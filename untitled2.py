import gym
import time
import pygame

"""
# blue : passenger
#puple : destination (dropout passenger)
#yellow/red : empty taxi
#green : full taxi
#RGBY destination,passenger location
"""


# Pygame'i başlat
pygame.init()


window_size = (350, 500)
window = pygame.display.set_mode(window_size)
pygame.display.set_caption("Taxi-v3 with Pygame")

env = gym.make("Taxi-v3", render_mode="rgb_array")
env.reset()

print(env.observation_space)
print(env.action_space)
state = env.encode(0,0,0,0)
print(state)
env.s = state

done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    
    # Ortamı renderla ve Pygame penceresine görüntüyü aktar
    frame = env.render()
    if frame is not None:
        frame = pygame.surfarray.make_surface(frame)
        window.blit(frame, (0, 0))
    
    pygame.display.update()
    time.sleep(0.1)  # Her frame arası biraz bekle
    
    # Rastgele bir eylem seç
    action = env.action_space.sample()
    
    # Yeni Gym sürümlerinde env.step(action) metodundan dönen değer sayısını kontrol et
    step_result = env.step(action)
    if len(step_result) == 4:
        obs, reward, done, info = step_result
    elif len(step_result) == 5:
        obs, reward, done, truncated, info = step_result
        done = done or truncated

env.close()
pygame.quit()
#%%
pygame.init()
env = gym.make("Taxi-v3",render_mode = "rgb_array")
env.reset()
print("State space",env.observation_space)



#Taksinin pozisyonu: 5x5 ızgara üzerinde 25 farklı pozisyon (0-24).
#Yolcunun pozisyonu: 5 farklı pozisyon (R, G, B, Y, takside).0-4 ile kodlanır.
#Yolcunun hedefi: 4 farklı hedef (R, G, B, Y). Bunlar sırasıyla 0-3 ile kodlanır.
#5*5*5*4 = 500
"""
There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
    Passenger locations:
  - 0: R(ed)
  - 1: G(reen)
  - 2: Y(ellow)
  - 3: B(lue)
  - 4: in taxi

  Destinations:
  - 0: R(ed)
  - 1: G(reen)
  - 2: Y(ellow)
  - 3: B(lue)
"""
print("Action space",env.action_space)
state = env.encode(3,1,2,2)
print("state number :",state)
env.s = state
env.render()

#%%
env.P[331]
"""
{0: [(1.0, 431, -1, False)],
 1: [(1.0, 231, -1, False)],
 2: [(1.0, 351, -1, False)],
 3: [(1.0, 331, -1, False)],
 4: [(1.0, 331, -10, False)],
 5: [(1.0, 331, -10, False)]}
""" 
#probability,next_state,reward,done
#0-5 action
# yanlış yerde pcikup(yolcu alma)veya pickoff yaparsak false 

#%%
env.reset()
time_step = 0
total_reward = 0
list_visualize = []
total_reward_list = []

for j in range(5):
    while True:
        time_step += 1
    
        # Choose action
        action = env.action_space.sample()  # Random action
    
        # Perform action and get reward
        results = env.step(action)
        print(len(results))  # Print the number of values returned
        print(results)  # Print the actual values returned
    
        # Assuming env.step(action) returns 4 values (state, reward, done, info)
        if len(results) == 4:
            state, reward, done, info = results
        else:
            # Adjust according to the number of values returned
            state = results[0]
            reward = results[1]
            done = results[2]
            info = results[3]
            # If there is an extra value
            extra_info = results[4] if len(results) > 4 else None
    
        # Measure (total) reward
        total_reward += reward
    
        # Visualize
        list_visualize.append({
            "frame": env.render(),
            "state": state,
            "action": action,
            "reward": reward,
            "total_reward": total_reward
        })
    
        #env.render()
    
        if done:
            total_reward_list.append(total_reward)
            break
#%%
import time
import matplotlib.pyplot as plt

for i, frame in enumerate(list_visualize):
    # frame["frame"] bir numpy.ndarray olduğundan, doğrudan görüntüleme veya analiz yapabilirsiniz.
    # Örneğin, görüntüyü göstermek için matplotlib kullanabilirsiniz.
    

    # Görüntüyü gösterme
    plt.imshow(frame["frame"])
    plt.title(f"Time_step: {i+1}")
    plt.show()
    
    # Bilgileri yazdırma
    print(f"Time_step: {i+1}")
    print(f"State: {frame['state']}")
    print(f"Action: {frame['action']}")
    print(f"Reward: {frame['reward']}")
    print(f"Total_reward: {frame['total_reward']}")
    print("\n\n")
    # Kısa bir süre bekleme
    #time.sleep(1)

#%%
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3",render_mode = None).env

# 1 - Q TABLE initiliaze
# 2 - for life or until learning stopped
# 3 - Choose an action (a) in current world state (s) based on 
# current Q value estimates
# 4- Take an action and observe the outcome state and reward
# 5- Update Q table



# Q TABLE initiliaze
state = env.observation_space.n # Discrete biçiminde .n ile numbere çevirdik
action = env.action_space.n # Discrete biçiminde .n ile numbere çevirdik
q_table = np.zeros([state,action]) 

# Hyperparametres
alpha = 0.1 # learning rate
gamma = 0.9
epsilon = 0.1 # %10 explore , %90 exploit

#Plotting Matrix
reward_list = []
dropouts_list = []


episode_number = 30000
for i in range(1,episode_number):
    #initialize enviroment
    state,_ = env.reset()
    # 1 epesiodelik kısım (müşteri yanlış kısıma bırakınca epesiode biter,
    #  yanana kadar birşeyler öğrenilir
    reward_count = 0
    dropouts = 0
    while True:
        
        #exploit(yerinde kalma) vs explore(keşfetme) to find action
        if random.uniform(0,1) < epsilon: # random sayı epsilondan küçükse action üret
            action = env.action_space.sample() # var olan actionlardan random bir action alma
        else:
            # q tableden en yüksek sayıda action yapan actionu seçmeliyiz
            # çünkü action rewarda göre seçiliyor ve daha iyi reward alması lazım
            action = np.argmax(q_table[state])
            #np.argmax en yüksek değerin indexini döndürüyor
              
        #action process and take reward/take observation
        next_state,reward,done,_,_= env.step(action) 
        
        # Q learning function
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state])
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        
        # Q table update 
        q_table[state,action] = next_value
        
        # Update state
        state = next_state
        
        # Find wrong dropouts
        if reward == -10:
            dropouts+=1
            
        if done: # done yanıp yanmadığımızı belirtecek
            break
        reward_count += reward
    if i%10 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode {},reward {},wrong droupouts {}".format(i,reward_count,dropouts))
        #zaman kaybından dolayı reward eksi çıkıyor

#%%
fig,axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropouts_list)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("dropouts")
axs[0].grid(True)
axs[1].grid(True)
plt.show()
#%%
import gym
import matplotlib.pyplot as plt
import numpy as np
"""There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger"""

# qtableda max değer hareket edilecek değerdir
env = gym.make("Taxi-v3")
env.reset()

state = env.encode(4,4,4,3) # burda 499 statede max değer 3.deki 17 değeridir yani batıya gitmemiz lazım

env.s = state



