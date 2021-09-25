import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # 增加传输的power
        self.power = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None
        # 传输的功率选择
        self.p = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        #self.max_speed = None
        #★★★★★★#
        self.max_speed = 60
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # 功率范围
        self.p_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # power contro dim
        self.dim_p_c = 1
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        #self.dt = 0.1
        #★★★★★★★★#
        self.dt = 0.5
        # ★★★★★★★★#
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # integrate physical state

        # update agent state
        for i,agent in enumerate(self.agents):
            self.integrate_position_state(agent,i)
            self.update_agent_state(agent)
            self.update_agent_power_state(agent)


    # integrate physical state
    def integrate_position_state(self,agent,i):
        #v_r = agent.action.u[0]*agent.max_speed
        #theta = agent.action.u[1]*3.14
        agent.state.p_vel[0] = agent.action.u[0]*0.707*agent.max_speed
        agent.state.p_vel[1] =  agent.action.u[1]*0.707*agent.max_speed
        agent.state.p_pos += agent.state.p_vel * self.dt
        #agent.state.p_pos = np.clip(agent.state.p_pos, -50, 50) # 1-11 clip the active area
        # 2021-1-12 23-44 clip region
        #agent.state.p_pos = np.clip(agent.state.p_pos, -500, 500) # 1-11 clip the active area
        # 2021-1-13 clip into the region
        agent.state.p_pos = np.clip(agent.state.p_pos, -500, 500) # 1-11 clip the active area
        #np.array([i // 2 * 500 - 250, i % 2 * 500 - 250], dtype=np.float)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def update_agent_power_state(self, agent):
        #agent.state.power = (agent.action.p + 1)/4 + 0.5
        agent.state.power = (agent.action.p + 1)/2
