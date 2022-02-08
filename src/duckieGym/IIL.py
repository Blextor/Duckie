class InteractiveImitationLearning:
    """
    A class used to contain main imitation learning algorithm
    ...
    Methods
    -------
    train(samples, debug)
        start training imitation learning
    """

    def __init__(self, env, teacher, learner, horizon, episodes, test=False):
        """
        Parameters
        ----------
        env :
            duckietown environment
        teacher :
            expert used to train imitation learning
        learner :
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """

        self.environment = env
        self.teacher = teacher
        self.learner = learner
        self.test = test

        # from IIL
        self._frame = horizon
        self._episodes = episodes

        # data
        self._observations = []
        self._expert_actions = []
        self.counter=0

        # statistics
        self.learner_action = None
        self.learner_uncertainty = None

        self.teacher_action = None
        self.active_policy = True  # if teacher is active

        # internal count
        self._current_frame = 0
        self._episode = 0

        # event listeners
        self._episode_done_listeners = []
        self._found_obstacle = False
        # steering angle gain
        self.gain = 10

    def train(self, debug=False):
        """
        Parameters
        ----------
        teacher :
            expert used to train imitation learning
        learner :
            model used to learn
        horizon : int
            which is the number of observations to be collected during one episode
        episode : int
            number of episodes which is the number of collected trajectories
        """
        self._debug = debug
        for episode in range(self._episodes):
            self._episode = episode
            self._sampling()
            self._optimize()  # episodic learning
            self._on_episode_done()


    def _sampling(self):

        observation = self.environment.render_obs() #get the first frame

        #for every frame in eposiode
        for frame in range(self._frame):
            self._current_frame = frame

            action = self._act(observation)

            try: #give the env the action and get back the state of the env
                next_observation, reward, done, info = self.environment.step(
                    [action[0], action[1] * self.gain]
                )
            except Exception as e:
                print(e)
            if self._debug: #TODO kell?
                self.environment.render()
            observation = next_observation

    # choose control policy and get the action
    def _act(self, observation):


        control_policy = self._mix()

        control_action = control_policy.predict(self.environment)

        self._query_expert(control_policy, control_action, self.environment)

        self.active_policy = (control_policy == self.teacher)
        if self.test:
            return self.learner_action

        return control_action


    #save the teachers actions and observations
    def _query_expert(self, control_policy, control_action, env):

        if control_policy == self.learner:
            self.learner_action = control_action
        else:
            self.learner_action = self.learner.predict(env)


        if control_policy == self.teacher:
            self.teacher_action = control_action
        else:
            self.teacher_action = self.teacher.predict(env)


        if self.teacher_action is not None:
            self._aggregate(env.obs, self.teacher_action)


        if self.teacher_action[0] < 0.1:
            self._found_obstacle = True
        else:
            self._found_obstacle = False

    def _mix(self): #to be implemented in derived classes
        raise NotImplementedError()

    def _aggregate(self, observation, action):
        if not (self.test):
            #if (self.counter<1):
            self._observations.append(observation)
            self._expert_actions.append(action)
            #self.counter+=1
            #if (self.counter>4):
             #   self.counter=0



    def on_episode_done(self, listener):
        self._episode_done_listeners.append(listener)

    def _on_episode_done(self):
        for listener in self._episode_done_listeners:
            listener.episode_done(self._episode)
        self.environment.reset()
