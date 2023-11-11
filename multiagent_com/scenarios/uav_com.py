import numpy as np
from multiagent_com.core import World, Agent, Landmark
from multiagent_com.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 4
        num_landmarks = 20
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 8
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            #agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = np.array([2/255.0, 162/255.0, 241/255.0])
            #agent.color = np.array([0.70, 0.83, 0.55])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            #landmark.color = np.array([248/255.0, 198/255.0, 149/255.0])

        # set random initial states
        agent_pos_lst = [np.array([-400, -400 ],dtype=np.float64),np.array([-400, 400 ],dtype=np.float64),
                         np.array([400, -400],dtype=np.float64),np.array([400, 400],dtype=np.float64)]
        for i, agent in enumerate(world.agents):
            # agent.state.p_pos = np.array([i//2*500-250, i%2*500-250 ],dtype=np.float64)
            agent.state.p_pos = agent_pos_lst[i]
            #agent.state.p_pos = np.random.uniform(-500, +500, world.dim_p)
            #np.random.uniform(-100, +100, world.dim_p)
            #agent.state.p_pos = np.array([-500, 500], dtype=np.float64)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.power = np.ones(world.dim_p_c) #np.zeros(world.dim_p_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-500, +500, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    # 用于最终统计实际表现的结果
    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        #print(agent1.state.p_pos)
        #print(agent2.state.p_pos)
        # 两者距离太近
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        '''rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1'''

        # First part
        # average and minimal data rate
        rew = 0
        num_agents = len(world.agents)
        num_landmarks = len(world.landmarks)
        serv_num = np.zeros(num_agents)
        serv_uav_idx = 100*np.ones(num_landmarks,dtype=int) # 每个user接入的uav的idx
        #serv_user_idx = np.zeros((num_agents,num_landmarks)) # 每个uav服务的用户idx
        for l_i, l in enumerate(world.landmarks):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            associ_uav_idx = np.argmin(dists) # 离l_i 最近的agent
            #if dists[associ_uav_idx] >360: # 加入接入范围限制 360更加合适
            #if dists[associ_uav_idx] >360 or serv_num[associ_uav_idx] >= 4: # 加入接入范围限制 360更加合适
            if dists[associ_uav_idx] >360: # 加入接入范围限制 360更加合适
                continue
            serv_num[associ_uav_idx] += 1  #associ_uav_idx 服务的用户数加1
            #serv_user_idx[associ_uav_idx,l_i] = 1 #associ_uav_idx服务用户l_i
            serv_uav_idx[l_i] = associ_uav_idx #用户l_i接入associ_uav_idx

        '''for l_i, l in enumerate(world.agents):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.landmarks]
            associ_user_idx = np.argmin(dists) # 离l_i 最近的agent
            if serv_num[l_i] == 1:
                continue
            serv_num[l_i] = 1  #associ_uav_idx 服务的用户数加1
            #serv_user_idx[associ_uav_idx,l_i] = 1 #associ_uav_idx服务用户l_i
            serv_uav_idx[associ_user_idx] = l_i #用户l_i接入associ_uav_idx'''

        # 500 mW
        noise = np.power(10, -100/10)
        P_ref = 500*np.power(10, -37.6/10) # -37.6 reference at 1m loss
        alpha = -2
        height = 100 #m
        p_los_B = 0.35
        p_los_C = 5
        #data_rate_requirement = 0.1 #np.array([0.1, 0.2, 0.3, 0.4])
        data_rate_requirement = 0.05  # np.array([0.1, 0.2, 0.3, 0.4])
        data_rate_list = np.zeros(num_landmarks)
        true_data_list = np.zeros(num_landmarks)
        lambda_ratio = 5
        P_blada_power = 0.012/8*1.225*0.05*0.79*np.power(400*0.5,3) # delta/8*rho*s*A*omega^3*R^3
        U_tip_speed_2 = 200*200
        trans_power = np.array([a.state.power for a in world.agents])
        velocity_list =  np.array([np.linalg.norm([a.state.p_vel[0] , a.state.p_vel[0] ]) for a in world.agents]) # p_vel[0] 是速度？
        for l_i, l in enumerate(world.landmarks):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)+np.square(height))) for a in world.agents]
            #dists = np.max(dists + [1])
            dists = np.clip(dists, 1, 1000) # np.max(dists + [1])
            serv_id = serv_uav_idx[l_i]
            if serv_id==100:
                #data_rate_list[l_i] = np.abs(10*(0 - data_rate_requirement[l_i]))
                #data_rate_list[l_i] = 1.0 /np.power( max(0/data_rate_requirement, 0.1),1)
                #data_rate_list[l_i] = float(0>=data_rate_requirement[l_i])
                #data_rate_list[l_i] = 10
                data_rate_list[l_i] = -1
                continue
            P_los_pro = 1.0 / (  1.0 + p_los_C* np.exp(-p_los_B*( 180/np.pi*np.arctan(height/dists) - p_los_C) )  )
            channel_loss = P_los_pro* np.power(10, -3/10) + (1-P_los_pro)* np.power(10, -23/10)
            signal = trans_power[serv_id]*P_ref*np.power(dists[serv_id],alpha)*channel_loss[serv_id]
            noise =  np.sum(trans_power*P_ref*np.power(dists,alpha)*channel_loss) - signal + noise
            #bandwidth =
            # last try to using TDMA
            #data_rate_value = np.log2(1+ signal/noise)/serv_num[serv_id]/num_agents
            data_rate_value = np.log2(1 + signal / noise) / serv_num[serv_id]
            true_data_list[l_i]  = data_rate_value
            #data_rate_list[l_i] = np.min( np.log2(1+ signal/noise),data_rate_requirement)
            #data_rate_list[l_i] = np.clip(data_rate_value, 0, data_rate_requirement)/data_rate_requirement
            #data_rate_list[l_i] = np.abs(10*(data_rate_value - data_rate_requirement))
            #data_rate_list[l_i] = 1.0 /np.power( max(data_rate_value/ data_rate_requirement, 0.1),1)
            data_rate_list[l_i] = float(data_rate_value>=data_rate_requirement)

        #power_consumption = trans_power + P_blada_power*(1+3.0*np.power(velocity_list,2)/U_tip_speed_2)
        power_consumption =  P_blada_power*(1+3.0*np.power(velocity_list,2)/U_tip_speed_2)
        power_consumption_min =  P_blada_power*(1+3.0*np.power(0,2)/U_tip_speed_2)
        power_consumption_max =  P_blada_power*(1+3.0*np.power(world.agents[0].max_speed,2)/U_tip_speed_2)
        #power_consumption = power_consumption / (1 + P_blada_power*(1+3.0*np.power(world.agents[0].max_speed,2)/U_tip_speed_2))
        power_consumption = (power_consumption - power_consumption_min)/(power_consumption_max - power_consumption_min)
        #rew += (np.sum(data_rate_list) - np.sum(power_consumption)*num_landmarks/num_agents)
        # last one added the power consumption
        #rew -= ( np.sum(data_rate_list) + np.sum(power_consumption)*0.1 )/(num_landmarks*num_landmarks/num_agents/np.sum(np.power(serv_num,2)) )
        rew += ( np.sum(data_rate_list) - 5*np.sum(power_consumption) - 0.1*np.sum(trans_power)  ) #/(num_landmarks*num_landmarks/num_agents/np.sum(np.power(serv_num,2)) )
        #rew += np.sum(data_rate_list)
        #rew -= ( np.sum(data_rate_list) )/(num_landmarks*num_landmarks/num_agents/np.sum(np.power(serv_num,2)) )
        '''print('user_satisfaction',data_rate_list)
        print('num_user_satisfaction', np.sum(data_rate_list) )
        print('tansmit_power',trans_power)
        print('velocity', velocity_list)
        print('power_consumption', power_consumption)'''
        world.data_rate_list = data_rate_list
        world.true_data_list = true_data_list
        world.trans_power = trans_power
        world.velocity_list = velocity_list
        world.power_consumption = power_consumption

        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def eval_data(self,world):
        #return (np.sum(world.data_rate_list), np.sum(world.true_data_list), world.trans_power,world.velocity_list)
        return (np.sum(world.data_rate_list), world.power_consumption, world.trans_power,world.velocity_list)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        range_limt = 500.0
        entity_pos = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        idx_nearby = np.argsort(dists)
        #idx_nearby = idx_nearby[:3]
        idx_nearby = idx_nearby[:3]
        landmarks_nearby = [world.landmarks[i] for i in idx_nearby]
        for entity in landmarks_nearby:  # world.entities:
            #entity_pos.append( entity.state.p_pos - agent.state.p_pos)
            entity_pos.append( (entity.state.p_pos - agent.state.p_pos)/range_limt)
        # entity colors
        '''entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)'''
        # communication of all other agents
        #comm = []
        '''other_pos = []
        for other in world.agents:
            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)'''

        other_pos = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.agents]
        idx_nearby = np.argsort(dists)
        idx_nearby = idx_nearby[:3 + 1]
        agents_nearby = [world.agents[i] for i in idx_nearby]
        for other in agents_nearby:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.state.p_pos - agent.state.p_pos)/range_limt)

        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + other_pos + entity_pos )
        return (np.concatenate([agent.state.p_vel/agent.max_speed] + [agent.state.p_pos/range_limt] + entity_pos + other_pos),0)