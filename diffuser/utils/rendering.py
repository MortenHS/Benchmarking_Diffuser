import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import ListedColormap
import gym
import mujoco_py as mjc
import warnings
import pdb
from .arrays import to_np
from .video import save_video, save_videos
from diffuser.datasets.d4rl import load_environment
from d4rl.locomotion.ant import AntMazeEnv

#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#
def env_map(env_name):
    '''
        map D4RL dataset names to custom fully-observed
        variants for rendering
    '''
    if 'halfcheetah' in env_name:
        return 'HalfCheetahFullObs-v2'
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
    else:
        return env_name
#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#
def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x
def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)
def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs
def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask
def plot2img(fig, remove_margins=True):
    # https://stackoverflow.com/a/35362787/2912349
    # https://stackoverflow.com/a/54334430/2912349
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))
#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#
class MuJoCoRenderer:
    '''
        default mujoco renderer
    '''
    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        ## @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
            self.viewer = None
    def pad_observation(self, observation):
        state = np.concatenate([
            np.zeros(1),
            observation,
        ])
        return state
    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        ## xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate([
            xpos[:,None],
            observations,
        ], axis=-1)
        return states
    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):
        if type(dim) == int:
            dim = (dim, dim)
        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)
        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 3,
                'lookat': [xpos, -0.5, 1],
                'elevation': -20
            }
        for key, val in render_kwargs.items():
            if key == 'lookat':
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)
        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation
        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])
        set_state(self.env, state)
        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data
    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)
    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False
        sample_images = self._renders(samples, partial=partial, **kwargs)
        composite = np.ones_like(sample_images[0]) * 255
        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]
        return composite
    def composite(self, savepath, paths, dim=(1024, 256), **kwargs):
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [5, 2, 0.5],
            'elevation': 0
        }
        images = []
        for path in paths:
            ## [ H x obs_dim ]
            path = atmost_2d(path)
            img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
            images.append(img)
        images = np.concatenate(images, axis=0)
        if savepath is not None:
            imageio.imsave(savepath, images)
            print(f'Saved {len(paths)} samples to: {savepath}')
        return images
    def render_rollout(self, savepath, states, **video_kwargs):
        if type(states) is list: states = np.array(states)
        images = self._renders(states, partial=True)
        save_video(savepath, images, **video_kwargs)
    def render_plan(self, savepath, actions, observations_pred, state, fps=30):
        ## [ batch_size x horizon x observation_dim ]
        observations_real = rollouts_from_state(self.env, state, actions)
        ## there will be one more state in `observations_real`
        ## than in `observations_pred` because the last action
        ## does not have an associated next_state in the sampled trajectory
        observations_real = observations_real[:,:-1]
        images_pred = np.stack([
            self._renders(obs_pred, partial=True)
            for obs_pred in observations_pred
        ])
        images_real = np.stack([
            self._renders(obs_real, partial=False)
            for obs_real in observations_real
        ])
        ## [ batch_size x horizon x H x W x C ]
        images = np.concatenate([images_pred, images_real], axis=-2)
        save_videos(savepath, *images)
    def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
        '''
            diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
        '''
        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
            'lookat': [10, 2, 0.5],
            'elevation': 0,
        }
        diffusion_path = to_np(diffusion_path)
        n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape
        frames = []
        for t in reversed(range(n_diffusion_steps)):
            print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')
            ## [ batch_size x horizon x observation_dim ]
            states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]
            frame = []
            for states in states_l:
                img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
                frame.append(img)
            frame = np.concatenate(frame, axis=0)
            frames.append(frame)
        save_video(savepath, frames, **video_kwargs)
    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)
#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#
MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12)
    # ,'antmaze-medium-diverse':(0, 9, 0, 12)
}

class MazeRenderer:
    def __init__(self, env):
        if type(env) is str: env = load_environment(env)
        self._config = env._config
        self._background = self._config != ' '
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(self._background * .5,
            extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)
        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0,1,path_length))
        plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
        plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img
    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
            savepath : str
            observations : [ n_paths x horizon x 2 ]
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'
        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)
        nrow = len(images) // ncol
        images = einops.rearrange(images,
            '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

class Maze2dRenderer(MazeRenderer):
    def __init__(self, env, observation_dim=None):
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)
    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]
        observations = observations + .5
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)

class AntMazeRenderer:
    def __init__(self, env, observation_dim=None):
        """
        Initialize the AntMazeRenderer with the given environment.

        Args:
            env (AntMazeEnv): The AntMaze environment instance.
            observation_dim (int, optional): Dimension of the observations.
        """
        self.env_name = env
        self.env = load_environment(env)
        self.observation_dim = np.prod(self.env.observation_space.shape) if observation_dim is None else observation_dim
        self.action_dim = np.prod(self.env.action_space.shape)
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.goal = self.env.goal_location if hasattr(self.env, 'goal_location') else None
        self._background = None

    def render(self, observations, conditions=None, **kwargs):
        """
        Render the trajectory of the agent in the AntMaze environment.

        Args:
            observations (np.ndarray): The agent's observations.
            conditions (np.ndarray, optional): Additional conditions, such as goals.
        """
        self.ax.clear()
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_aspect('equal', adjustable='box')

        # Draw the maze layout if available
        if hasattr(self.env, 'maze_arr'):
            self._render_maze(self.env.maze_arr)

        # Draw goal location if available
        if self.goal is not None:
            self.ax.add_patch(Circle(self.goal, radius=0.2, color='green', label='Goal'))

        # Plot the observations
        for obs in observations:
            self.ax.plot(obs[0], obs[1], 'ro', markersize=3)

        plt.draw()
        plt.pause(0.01)

    def _render_maze(self, maze_arr):
        """
        Render the maze layout using the provided maze array.
        """
        for i, row in enumerate(maze_arr):
            for j, cell in enumerate(row):
                if cell == 1:  # Wall
                    self.ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))
    
    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
        Generate a composite image of multiple trajectories.

        Args:
            savepath (str): Path to save the output image.
            paths (list): List of paths, where each path is [horizon x 2].
            ncol (int): Number of columns in the composite image.
        '''
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by the number of columns'
        images = []

        # Iterate over each path and render it using the AntMazeRenderer
        for path in paths:
            # Create a temporary list to store rendered frames for the current path
            for obs in path:
                # Render the current observation and convert the figure to an image array
                self.render([obs])
                
                # Capture the current plot as an image array
                self.fig.canvas.draw()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                
                images.append(img)

        # Stack images and arrange them into a grid
        images = np.stack(images, axis=0)
        nrow = len(images) // ncol
        images = einops.rearrange(images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        
        # Save the composite image
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')


    def close(self):
        """
        Close the rendering window.
        """
        plt.close(self.fig)

# class AntMazeRenderer(MazeRenderer):
    
#     def __init__(self, env, observation_dim=None):
#         self.env_name = env
#         self.env = load_environment(env)
#         self.observation_dim = np.prod(self.env.observation_space.shape)
#         self.action_dim = np.prod(self.env.action_space.shape)
#         self.goal = None #self.env.target_goal  # Set goal position based on AntMaze target
#         self._background = self.env.maze_arr # == 10
#         self._remove_margins = False
#         self._extent = (0, 1, 1, 0)
#     def renders(self, observations, conditions=None, **kwargs):
#         # Adjust observations based on environment bounds, if any
#         bounds = MAZE_BOUNDS.get(self.env_name)
        
#         observations = observations + 0.5
#         if len(bounds) == 2:
#             _, scale = bounds
#             observations /= scale
#         elif len(bounds) == 4:
#             _, iscale, _, jscale = bounds
#             observations[:, 0] /= iscale
#             observations[:, 1] /= jscale
#         else:
#             raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
#         if conditions is not None:
#             conditions /= scale
#         return super().renders(observations, conditions, **kwargs)


#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#
def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
            f'but got state of size {state.size}')
        state = state[:qpos_dim + qvel_dim]
    env.set_state(state[:qpos_dim], state[qpos_dim:])
def rollouts_from_state(env, state, actions_l):
    rollouts = np.stack([
        rollout_from_state(env, state, actions)
        for actions in actions_l
    ])
    return rollouts

def rollout_from_state(env, state, actions):
    qpos_dim = env.sim.data.qpos.size
    env.set_state(state[:qpos_dim], state[qpos_dim:])
    observations = [env._get_obs()]
    for act in actions:
        obs, rew, term, _ = env.step(act)
        observations.append(obs)
        if term:
            break
    for i in range(len(observations), len(actions)+1):
        ## if terminated early, pad with zeros
        observations.append( np.zeros(obs.size) )
    return np.stack(observations)