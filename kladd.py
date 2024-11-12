class AntMazeEnv(maze_env.MazeEnv, GoalReachingAntEnv, offline_env.OfflineEnv):
    """Ant navigating a maze."""
    LOCOMOTION_ENV = GoalReachingAntEnv

    def __init__(self, 
                 goal_sampler=None, 
                 expose_all_qpos=True,
                 reward_type='dense', 
                 v2_resets=False, 
                 maze_arr=None,
                 str_maze_spec=MEDIUM_MAZE,
                 *args, 
                 **kwargs):
        # Initialize goal_sampler if not provided
        if goal_sampler is None:
            goal_sampler = lambda np_rand: maze_env.MazeEnv.goal_sampler(self, np_rand)
        
        # Initialize the parent MazeEnv
        maze_env.MazeEnv.__init__(
            self, *args, manual_collision=False,
            goal_sampler=goal_sampler,
            expose_all_qpos=expose_all_qpos,
            reward_type=reward_type,
            **kwargs
        )
        
        # Initialize OfflineEnv
        offline_env.OfflineEnv.__init__(self, **kwargs)

        # Set up the maze array if provided, otherwise generate from str_maze_spec
        if maze_arr is not None:
            self.maze_arr = maze_arr
        elif str_maze_spec is not None:
            self.maze_arr = parse_maze(str_maze_spec)
        else:
            raise ValueError("Either maze_arr or str_maze_spec must be provided")

        # Log the initial maze specifications
        self.str_maze_spec = str_maze_spec
        
        # Initialize observation logging for rendering
        self._observations = []

        # Set the initial target goal
        self.set_target()
        self.v2_resets = v2_resets

    def reset(self):
        """Resets the environment and sets a new target if v2_resets is enabled."""
        if self.v2_resets:
            self.set_target()
        obs = super().reset()
        
        # Clear the observation log and record the initial position
        self._observations = [self.get_agent_position()]
        return obs

    def step(self, action):
        """Perform a step in the environment and record the ant's position."""
        obs, reward, done, info = super().step(action)
        
        # Log the agent's current position for rendering
        self._observations.append(self.get_agent_position())
        return obs, reward, done, info

    def get_agent_position(self):
        """Returns the current (x, y) position of the agent."""
        torso = self.sim.data.get_body_xpos("torso")
        return np.array([torso[0], torso[1]])

    def set_target(self, target_location=None):
        """Set a new target location."""
        return self.set_target_goal(target_location)

    def seed(self, seed=0):
        """Sets the seed for the environment."""
        mujoco_env.MujocoEnv.seed(self, seed)

# Antmazerenderer forslag fra chat.

class AntMazeRenderer:
    def __init__(self, env):
        """Initialize the renderer with the environment."""
        self.env = env
        self.maze_arr = env.maze_arr
        self._background = (self.maze_arr == 10)  # Render walls
        self._extent = (0, self.maze_arr.shape[1], self.maze_arr.shape[0], 0)

    def render_trajectory(self, observations, title=None):
        """
        Renders the maze with the trajectory.
        observations: list of (x, y) positions of the ant.
        """
        plt.clf()
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 7)

        # Draw the maze background
        ax.imshow(self._background, cmap='binary', extent=self._extent, vmin=0, vmax=1)

        # Plot the trajectory
        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0, 1, path_length))

        # Convert observations to maze coordinates (scale to fit)
        observations = np.array(observations)
        ax.plot(observations[:, 1], observations[:, 0], c='black', zorder=10)
        ax.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)

        # Draw the goal position if available
        if self.env.target_goal is not None:
            goal_pos = self.env.target_goal
            ax.scatter(goal_pos[1], goal_pos[0], c='red', marker='*', s=100, label='Goal', zorder=30)

        plt.axis('off')
        if title:
            plt.title(title)

        plt.show()

    def render_frame(self, observation, title=None):
        """Render a single frame of the maze with the current position."""
        self.render_trajectory([observation], title=title)

    def composite(self, savepath, paths, ncol=5, **kwargs):
        """Generate a composite image of multiple trajectories."""
        assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'
        images = []
        for path in paths:
            img = self.render_trajectory(path)
            images.append(img)

        images = np.stack(images, axis=0)
        nrow = len(images) // ncol
        images = einops.rearrange(images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

# class AntMazeRenderer:
#     def __init__(self, env, observation_dim=None):
#         """
#         Initialize the AntMazeRenderer with the given environment.

#         Args:
#             env (AntMazeEnv): The AntMaze environment instance.
#             observation_dim (int, optional): Dimension of the observations.
#         """
#         self.env_name = env
#         self.env = load_environment(env)
#         self.observation_dim = np.prod(self.env.observation_space.shape) if observation_dim is None else observation_dim
#         self.action_dim = np.prod(self.env.action_space.shape)
#         self.fig, self.ax = plt.subplots()
#         self.ax.set_xlim(-5, 5)
#         self.ax.set_ylim(-5, 5)
#         self.goal = self.env.goal_location if hasattr(self.env, 'goal_location') else None
#         self._background = self.env.maze_arr == 10

#     def render(self, observations, conditions=None, **kwargs):
#         """
#         Render the trajectory of the agent in the AntMaze environment.

#         Args:
#             observations (np.ndarray): The agent's observations.
#             conditions (np.ndarray, optional): Additional conditions, such as goals.
#         """
#         self.ax.clear()
#         self.ax.set_xlim(-5, 5)
#         self.ax.set_ylim(-5, 5)
#         self.ax.set_aspect('equal', adjustable='box')

#         # Draw the maze layout if available
#         if hasattr(self.env, 'maze_arr'):
#             self._render_maze(self.env.maze_arr)

#         # Draw goal location if available
#         if self.goal is not None:
#             self.ax.add_patch(Circle(self.goal, radius=0.2, color='green', label='Goal'))

#         # Plot the observations
#         for obs in observations:
#             self.ax.plot(obs[0], obs[1], 'ro', markersize=3)

#         plt.draw()
#         plt.pause(0.01)

#     def _render_maze(self, maze_arr):
#         """
#         Render the maze layout using the provided maze array.
#         """
#         for i, row in enumerate(maze_arr):
#             for j, cell in enumerate(row):
#                 if cell == 1:  # Wall
#                     self.ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))
    
    # def composite(self, savepath, paths, ncol=5, **kwargs):
    #     '''
    #         savepath : str
    #         observations : [ n_paths x horizon x 2 ]
    #     '''
    #     assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'
    #     images = []
    #     for path, kw in zipkw(paths, **kwargs):
    #         img = self.render(*path, **kw) #Originally renders with an s
    #         images.append(img)
    #     images = np.stack(images, axis=0)
    #     nrow = len(images) // ncol
    #     images = einops.rearrange(images,
    #         '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
    #     imageio.imsave(savepath, images)
    #     print(f'Saved {len(paths)} samples to: {savepath}')

    def composite(self, savepath, paths, ncol=5, **kwargs):
        '''
        Generate a composite image of multiple trajectories.
        Args:
            savepath (str): Path to save the output image.
            paths (list): List of paths, where each path is [horizon x 2].
            ncol (int): Number of columns in the composite image.
        '''
        assert len(paths) % ncol == 0
        images = []
        # Iterate over each path and render it using the AntMazeRenderer
        for path in paths:
            # Render the trajectory and capture the output as an image
            for obs in path:
                # Render the current observation and convert it to an image
                self.render([obs])
                self.fig.canvas.draw()
        
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                
                if len(img.shape) == 3:
                    images.append(img)
        
        if len(images) == 0:
            raise ValueError("No images were rendered.")
        
        images = np.stack(images, axis=0)
        if len(images.shape) != 4:
            raise ValueError(f"Expected 4-dimensional array, but got {images.shape}")
        
        nrow = len(images) // ncol
        images = einops.rearrange(
            images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol
        )
        
        # Save the composite image
        imageio.imsave(savepath, images)
        print(f'Saved {len(paths)} samples to: {savepath}')

    # def composite(self, savepath, paths, ncol=5, **kwargs):
    #     '''
    #     Generate a composite image of multiple trajectories.

    #     Args:
    #         savepath (str): Path to save the output image.
    #         paths (list): List of paths, where each path is [horizon x 2].
    #         ncol (int): Number of columns in the composite image.
    #     '''
    #     assert len(paths) % ncol == 0, 'Number of paths must be divisible by the number of columns'
    #     images = []

    #     # Iterate over each path and render it using the AntMazeRenderer
    #     for path in paths:
    #         # Create a temporary list to store rendered frames for the current path
    #         for obs in path:
    #             # Render the current observation and convert the figure to an image array
    #             self.render([obs])
                
    #             # Capture the current plot as an image array
    #             self.fig.canvas.draw()
    #             img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                
    #             images.append(img)

    #     # Stack images and arrange them into a grid
    #     images = np.stack(images, axis=0)
    #     nrow = len(images) // ncol
    #     images = einops.rearrange(images, '(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
        
    #     # Save the composite image
    #     imageio.imsave(savepath, images)
    #     print(f'Saved {len(paths)} samples to: {savepath}')

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