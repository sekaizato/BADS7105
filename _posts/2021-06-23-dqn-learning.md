---
title: Deep Q Learning (DQN) Reinforcement
date: 2021-06-23 00:00:00
description: Practice Deep Q Learning from Machine Learning for Trading Course by Google on Coursera.
featured_image: '/images/demo.jpg'
---
{% comment %}
    {% raw %}

```python
PROJECT_LINK = 'dqn-learning'
# PATH = '/Users/touchpadthamkul/zatoDev/clone/dev'
PATH = ''


# FRAMEWORK
from IPython.display import Markdown as md
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import datetime, pytz
import numpy as np
import os

pio.renderers.default = 'colab'

def getVariableNames(variable):
    results = []
    globalVariables=globals().copy()
    for globalVariable in globalVariables:
        if id(variable) == id(globalVariables[globalVariable]):
            results.append(globalVariable)
    return results

def displayPlot(fig):
    project_id = PROJECT_LINK.replace(' ','_')
    fig_json = fig.to_json()
    fig_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date())+'-'+project_id+'_'+getVariableNames(fig)[0]
    filename = fig_name+'.html'
    if PATH != '':
        save_path = PATH + '/_includes/post-figures/'
    else:
        save_path = ''
    completeName = os.path.join(save_path, filename)
    template = """
<html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id='{1}'></div>
        <script>
            var plotly_data = {0};
            let config = {{displayModeBar: false }};
            Plotly.react('{1}', plotly_data.data, plotly_data.layout, config);
        </script>
    </body>
</html>
"""
    # write the JSON to the HTML template
    with open(completeName, 'w') as f:
        f.write(template.format(fig_json, fig_name))
    return md("{% include post-figures/" + filename + " full_width=true %}")

def displayImg(img_name):
    master_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date()) + '-' + PROJECT_LINK + '-' + img_name
    !cp -frp $img_name $master_name
    if PATH != '':     
        img_path = PATH + '/images/projects'
        !mv $master_name $img_path
        output = md("![](/images/projects/" + master_name +")")        
    else:
        img_path = PATH
        output = md("![]("+master_name +")")
    return output

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def runBrowser(url):
    url = 'https://zato.dev/blog/' + PROJECT_LINK
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("window-size=375,812")
    # browser = webdriver.Chrome('/Users/touchpadthamkul/PySelenium/chromedriver', chrome_options=chrome_options)
    browser = webdriver.Chrome(ChromeDriverManager().install(),chrome_options=chrome_options)
    browser.get(url)

    
import ipynbname

def saveExport():        
    pynb_name = ipynbname.name() +'.ipynb'
    md_name = ipynbname.name() +'.md'
    if PATH != '':
        selected = int(input('1 posts \n2 projects\n'))
        if selected != 1:
            folder = '/_projects'
        else:
            folder = '/_posts'
        post_path = PATH + folder
    else:
        post_path = ''
    master_name = str(datetime.datetime.now(tz=pytz.timezone('Asia/Bangkok')).date()) + '-' + PROJECT_LINK + '.md'
    !jupyter nbconvert --to markdown $pynb_name
    !mv $md_name $master_name
    !mv $master_name $post_path

# saveExport()
# runBrowser(url)
```
    {% endraw %}
{% endcomment %}
## Intro

We'll Simulation environment that will Game Over with 100 Rounds
The Game is considered "won" when the pole can stay up for an average of steps 195 over 100 games.

An agent

### Import Library


```python
from collections import deque
import random

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
```

### Create Environment


```python
env = gym.make('CartPole-v0')

print(f'The Observation Space is {env.observation_space}')
print(f'The Observation Dimensions are {env.observation_space.shape}')

print(f'The action space is {env.action_space}')
print(f'The number of possible actions is {env.action_space.n}')
```

    The Observation Space is Box(4,)
    The Observation Dimensions are (4,)
    The action space is Discrete(2)
    The number of possible actions is 2


### State Information
Let's check array of state in environment. First we'll reset all information and print state of environment


```python
for i in range(0,3):
    state = env.reset()
    print(state)
```

    [ 0.03429787 -0.01420067 -0.01842509 -0.00689476]
    [ 0.01719735  0.02225807  0.04263482 -0.03059391]
    [ 0.04199883 -0.01596742 -0.02380738  0.03253764]


> State of Deep RN are Multidimensional so we can Add Another Dimansion in the future

Print State Function for recheck


```python
def print_state(state, step, reward=None):
    format_string = 'Step {0} - Cart X: {1:.3f}, Cart V: {2:.3f}, Pole A: {3:.3f}, Pole V:{4:.3f}, Reward: {5}'
    print(format_string.format(step, *tuple(state), reward))
    
state = env.reset()
step = 0
print_state(state, step)
```

    Step 0 - Cart X: 0.040, Cart V: -0.004, Pole A: -0.001, Pole V:0.028, Reward: None


Next we'll test the action by given input to environment <br>
( Let's 0 = left , 1 = right )


```python
# Move Left
action = 0
state_prime, reward, done, info = env.step(action)
step += 1
print_state(state_prime, step, reward)
print("The Game is over." if done else "The game can continue")
print("Info: ", info)
```

    Step 1 - Cart X: 0.040, Cart V: -0.199, Pole A: 0.000, Pole V:0.321, Reward: 1.0
    The game can continue
    Info:  {}


Move untile the game is over


```python
action = 1
state_prime, reward, done, info = env.step(action)
step += 1

print_state(state_prime, step, reward)
print("The Game is over." if done else "The game can continue")
print("Info: ", info)
```

    Step 15 - Cart X: 0.340, Cart V: 2.535, Pole A: -0.462, Pole V:-4.178, Reward: 0.0
    The Game is over.
    Info:  {}



```python
actions = [x % 2 for x in range(200)]
state = env.reset()
step = 0
episode_reward = 0 
done = False
while not done and step < len(actions):
    action = actions[step]
    state_prime, reward, done, info = env.step(action)
    episode_reward += reward
    step += 1
    state = state_prime
    print_state(state, step, reward)

end_statement = "Game over !" if done else "Ran out of actions!"
print(end_statement, "Score =", episode_reward)
```

    Step 1 - Cart X: -0.046, Cart V: -0.153, Pole A: 0.038, Pole V:0.307, Reward: 1.0
    Step 2 - Cart X: -0.049, Cart V: 0.042, Pole A: 0.044, Pole V:0.026, Reward: 1.0
    Step 3 - Cart X: -0.048, Cart V: -0.154, Pole A: 0.044, Pole V:0.332, Reward: 1.0
    Step 4 - Cart X: -0.051, Cart V: 0.041, Pole A: 0.051, Pole V:0.054, Reward: 1.0
    Step 5 - Cart X: -0.051, Cart V: -0.155, Pole A: 0.052, Pole V:0.362, Reward: 1.0
    Step 6 - Cart X: -0.054, Cart V: 0.039, Pole A: 0.059, Pole V:0.086, Reward: 1.0
    Step 7 - Cart X: -0.053, Cart V: -0.157, Pole A: 0.061, Pole V:0.397, Reward: 1.0
    Step 8 - Cart X: -0.056, Cart V: 0.037, Pole A: 0.069, Pole V:0.124, Reward: 1.0
    Step 9 - Cart X: -0.055, Cart V: -0.159, Pole A: 0.071, Pole V:0.438, Reward: 1.0
    Step 10 - Cart X: -0.058, Cart V: 0.035, Pole A: 0.080, Pole V:0.169, Reward: 1.0
    Step 11 - Cart X: -0.058, Cart V: -0.161, Pole A: 0.084, Pole V:0.485, Reward: 1.0
    Step 12 - Cart X: -0.061, Cart V: 0.033, Pole A: 0.093, Pole V:0.220, Reward: 1.0
    Step 13 - Cart X: -0.060, Cart V: -0.163, Pole A: 0.098, Pole V:0.541, Reward: 1.0
    Step 14 - Cart X: -0.064, Cart V: 0.030, Pole A: 0.108, Pole V:0.280, Reward: 1.0
    Step 15 - Cart X: -0.063, Cart V: -0.166, Pole A: 0.114, Pole V:0.605, Reward: 1.0
    Step 16 - Cart X: -0.066, Cart V: 0.027, Pole A: 0.126, Pole V:0.351, Reward: 1.0
    Step 17 - Cart X: -0.066, Cart V: -0.169, Pole A: 0.133, Pole V:0.680, Reward: 1.0
    Step 18 - Cart X: -0.069, Cart V: 0.024, Pole A: 0.147, Pole V:0.432, Reward: 1.0
    Step 19 - Cart X: -0.069, Cart V: -0.173, Pole A: 0.155, Pole V:0.767, Reward: 1.0
    Step 20 - Cart X: -0.072, Cart V: 0.019, Pole A: 0.171, Pole V:0.527, Reward: 1.0
    Step 21 - Cart X: -0.072, Cart V: -0.178, Pole A: 0.181, Pole V:0.869, Reward: 1.0
    Step 22 - Cart X: -0.075, Cart V: 0.015, Pole A: 0.199, Pole V:0.638, Reward: 1.0
    Step 23 - Cart X: -0.075, Cart V: -0.183, Pole A: 0.211, Pole V:0.986, Reward: 1.0
    Game over ! Score = 23.0


It's a challenge to get to 200. We could repeatedly experiment to find the best heuristics to beat the game, or we could leave all that work to the robot. Let's create an intellicene to figure this out for us

## Bulding the Agent

We take in both state and actions as inputs to our network. The states are fed in as normal, but the actions are used to 'mask' the output

The Bellman Equation actually isn't in the network. That's because this is only the 'brain' of our agent.

Just like other neural network algorithms ,we need data to train on. However, this data is the result of our simulations, not something previously stored in a table we're going to give our agent a memory where we can store state - action new state transitions to learn on

Each time the agent takes a step in gym, we'll save (state, action, reward, state_prime, done) to our buffer


```python
class Memory():
    
    # Define Parameter on Memory
    def __init__(self, memory_size, batch_size, gamma):
        self.buffer = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=self.batch_size, replace=False)
        
        batch = np.array([self.buffer[i] for i in index]).T.tolist()
        states_mb = tf.convert_to_tensor(np.array(batch[0], dtype=np.float32))
        actions_mb = np.array(batch[1], dtype=np.int8)
        rewards_mb = np.array(batch[2], dtype=np.float32)
        states_prime_mb = np.array(batch[3], dtype=np.float32)
        dones_mb = batch[4]
        return states_mb, actions_mb, rewards_mb, states_prime_mb, dones_mb
        
test_memory_size = 20
test_batch_size = 4
test_gamma = .9

test_memory = Memory(test_memory_size, test_batch_size, test_gamma)
```


```python
actions = [x % 2 for x in range(200)]
state = env.reset()
step = 0
episode_reward = 0
done = False

while not done and step < len(actions):
    action = actions[step]
    state_prime, reward, done, info = env.step(action)
    episode_reward += reward
    test_memory.add((state, action, reward, state_prime, done))
    step += 1
    state = state_prime
    print_state(state, step, reward)

end_statement = "Game over!" if done else "Ran out of actions!"
print(end_statement, "Score =", episode_reward)
```

    Step 1 - Cart X: -0.026, Cart V: -0.192, Pole A: 0.010, Pole V:0.279, Reward: 1.0
    Step 2 - Cart X: -0.030, Cart V: 0.003, Pole A: 0.016, Pole V:-0.010, Reward: 1.0
    Step 3 - Cart X: -0.030, Cart V: -0.192, Pole A: 0.016, Pole V:0.287, Reward: 1.0
    Step 4 - Cart X: -0.034, Cart V: 0.003, Pole A: 0.021, Pole V:-0.000, Reward: 1.0
    Step 5 - Cart X: -0.034, Cart V: -0.193, Pole A: 0.021, Pole V:0.299, Reward: 1.0
    Step 6 - Cart X: -0.037, Cart V: 0.002, Pole A: 0.027, Pole V:0.013, Reward: 1.0
    Step 7 - Cart X: -0.037, Cart V: -0.193, Pole A: 0.028, Pole V:0.314, Reward: 1.0
    Step 8 - Cart X: -0.041, Cart V: 0.001, Pole A: 0.034, Pole V:0.030, Reward: 1.0
    Step 9 - Cart X: -0.041, Cart V: -0.194, Pole A: 0.034, Pole V:0.334, Reward: 1.0
    Step 10 - Cart X: -0.045, Cart V: 0.000, Pole A: 0.041, Pole V:0.052, Reward: 1.0
    Step 11 - Cart X: -0.045, Cart V: -0.195, Pole A: 0.042, Pole V:0.357, Reward: 1.0
    Step 12 - Cart X: -0.049, Cart V: -0.001, Pole A: 0.049, Pole V:0.078, Reward: 1.0
    Step 13 - Cart X: -0.049, Cart V: -0.197, Pole A: 0.051, Pole V:0.386, Reward: 1.0
    Step 14 - Cart X: -0.053, Cart V: -0.002, Pole A: 0.059, Pole V:0.110, Reward: 1.0
    Step 15 - Cart X: -0.053, Cart V: -0.198, Pole A: 0.061, Pole V:0.420, Reward: 1.0
    Step 16 - Cart X: -0.057, Cart V: -0.004, Pole A: 0.069, Pole V:0.147, Reward: 1.0
    Step 17 - Cart X: -0.057, Cart V: -0.200, Pole A: 0.072, Pole V:0.461, Reward: 1.0
    Step 18 - Cart X: -0.061, Cart V: -0.006, Pole A: 0.081, Pole V:0.192, Reward: 1.0
    Step 19 - Cart X: -0.061, Cart V: -0.202, Pole A: 0.085, Pole V:0.509, Reward: 1.0
    Step 20 - Cart X: -0.065, Cart V: -0.008, Pole A: 0.095, Pole V:0.245, Reward: 1.0
    Step 21 - Cart X: -0.065, Cart V: -0.205, Pole A: 0.100, Pole V:0.566, Reward: 1.0
    Step 22 - Cart X: -0.070, Cart V: -0.011, Pole A: 0.112, Pole V:0.306, Reward: 1.0
    Step 23 - Cart X: -0.070, Cart V: -0.208, Pole A: 0.118, Pole V:0.632, Reward: 1.0
    Step 24 - Cart X: -0.074, Cart V: -0.014, Pole A: 0.130, Pole V:0.379, Reward: 1.0
    Step 25 - Cart X: -0.074, Cart V: -0.211, Pole A: 0.138, Pole V:0.709, Reward: 1.0
    Step 26 - Cart X: -0.078, Cart V: -0.018, Pole A: 0.152, Pole V:0.463, Reward: 1.0
    Step 27 - Cart X: -0.079, Cart V: -0.215, Pole A: 0.161, Pole V:0.800, Reward: 1.0
    Step 28 - Cart X: -0.083, Cart V: -0.022, Pole A: 0.177, Pole V:0.562, Reward: 1.0
    Step 29 - Cart X: -0.084, Cart V: -0.220, Pole A: 0.189, Pole V:0.905, Reward: 1.0
    Step 30 - Cart X: -0.088, Cart V: -0.027, Pole A: 0.207, Pole V:0.677, Reward: 1.0
    Step 31 - Cart X: -0.088, Cart V: -0.225, Pole A: 0.220, Pole V:1.027, Reward: 1.0
    Game over! Score = 31.0


Now, let's sample the memory by running the cell below multiple times. It's different each call, and that's on purpose

Just like with other neural networks, it's important to randomly sample so that our agent can learn from many different situations

The use of a memory buffer is called Experience Replay.

> Experience Replay is a technique of a uniform random sample.


```python
test_memory.sample()
```

    <ipython-input-195-efd76fe107f0>:16: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      batch = np.array([self.buffer[i] for i in index]).T.tolist()





    (<tf.Tensor: shape=(4, 4), dtype=float32, numpy=
     array([[-0.07393754, -0.01437452,  0.13038343,  0.37862304],
            [-0.0784467 , -0.01811479,  0.15214403,  0.46313375],
            [-0.04518198, -0.19538692,  0.04217282,  0.35729435],
            [-0.05308751, -0.19822866,  0.06079957,  0.42039365]],
           dtype=float32)>,
     array([0, 0, 1, 1], dtype=int8),
     array([1., 1., 1., 1.], dtype=float32),
     array([[-0.07422502, -0.21108375,  0.13795589,  0.7094067 ],
            [-0.07880899, -0.21502253,  0.16140671,  0.79964143],
            [-0.04908972, -0.00088912,  0.0493187 ,  0.07820219],
            [-0.05705208, -0.0040185 ,  0.06920744,  0.14748076]],
           dtype=float32),
     [False, False, False, False])



But before the agent has any memories and has learned anything, how is it supposed to act? That comes down to Exploration vs Exploitation.

The trouble is that in order to learn, risks with t


```python
class Partial_Agent():
    def __init__(self, network, memory, epsilon_decay, action_size):
        self.network = network
        self.action_size = action_size
        self.memory = memory
        self.epsilon = 1  # The chance to take a random action
        self.epsilon_decay = epsilon_decay
    
    def act(self, state, training=False):
        if training:
            if len(self.memory.buffer) >= self.memory.batch_size:
                self.epsilon *= self.epsilon_decay
            if self.epsilon > np.random.rand():
                print('Exploration!')
                return random.randint(0, self.action_size-1)
        
        print('Exploitation!')
        state_batch = np.expand_dims(state, axis=0)
        predict_mask = np.ones((1, self.action_size,))
        action_qs = self.network.predict([state_batch, predict_mask])
        return np.argmax(action_qs[0])
    

```

Let's define the agent and get a starting state to see how it would act without any training


```python
def deep_q_network(state_shape, action_size, learning_rate, hidden_neurons):
    state_input = layers.Input(state_shape, name='frames')
    actions_input = layers.Input((action_size,), name='mask')
    
    hidden_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
    hidden_2 = layers.Dense(hidden_neurons, activation='relu')(hidden_1)
    q_values = layers.Dense(action_size)(hidden_2)
    masked_q_values = layers.Multiply()([q_values, actions_input])
    
    model = models.Model(
    inputs=[state_input, actions_input], outputs=masked_q_values)
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    model.compile(loss='mse', optimizer=optimizer)
    return model
```


```python
state = env.reset()

space_shape = env.observation_space.shape
action_size = env.action_space.n

test_learning_rate = .2
test_hidden_neurons = 10
test_epsilon_decay = .95

test_network = deep_q_network(space_shape, action_size, test_learning_rate, test_hidden_neurons)

test_agent = Partial_Agent(test_network, test_memory, test_epsilon_decay, action_size)
```


```python
action = test_agent.act(state, training=True)
print('Push Right' if action else "Push Left")
```

    Exploration!
    Push Left


Memories, a brain, and a healthy dose of curiosity. We finally have all the ingredient for our agent to learn.

Below is the code used by our agent to learn, where the Bellman Equation at last make an appearance. We'll run through the folowing steps

1. Pull a batch from memory
2. Get Q value (based on memory's ending state
* Assume the Q value of the action with the highest Q value (test all actions)
3. Update these Q values with Bellman Equation
* target_qs = (next_q_mb * self.memory.gamma) + reward_mb
* If the state is the end of the game, set the target_q to the reward for entering the final state
4. Reshape the target_qs to match the networks output
* Only learn on the memory's corresponding action by setting all action nodes to zero besides the action node taken.
5. Fit Target Qs as the label to our model against the memory's starting state and action as the inputs


```python
def learn(self):
    batch_size = self.memory.batch_size
    if len(self.memory.buffer) < batch_size:
        return None
    
    # Obtain random mini-batch from memory
    state_mb, action_mb, reward_mb, next_state_mb, done_mb = (self.memory.sample())
    
    # Get Q values for next_state
    predict_mask = np.ones(action_mb.shape + (self.action_size,))
    next_q_mb = self.network.predict([next_state_mb, predict_mask])
    next_q_mb = tf.math.reduce_max(next_q_mb, axis=1)
    
    # Apply the Bellman Equation
    target_qs = (next_q_mb * self.memory.gamma) + reward_mb
    target_qs = tf.where(done_mb, reward_mb, target_qs)
    
    # Match training batch to network output:
    # Target_q where action taken, 0 otherwise.
    
    action_mb = tf.convert_to_tensor(action_mb)
    action_hot = tf.one_hot(action_mb, self.action_size)
    target_mask = tf.mulltiply(tf.expand_dims(target_qs, -1), action_hot)
    
    return self.network.train_on_batch([state_mb, action_hot], target_mask, reset_metrics=False)

Partial_Agent.learn = learn
test_agent = Partial_Agent(test_network, test_memory, test_epsilon_decay, action_size)
```


```python
state = env.reset()
step = 0
episode_reward = 0
done = False

while not done:
    action = test_agent.act(state, training=True)
    state_prime, reward, done, info = env.step(action)
    episode_reward += reward
    test_agent.memory.add((state, action, reward, state_prime, done))
    step += 1
    state = state_prime
    print_state(state, step, reward)

```

    Exploitation!
    Step 1 - Cart X: -0.032, Cart V: -0.152, Pole A: 0.027, Pole V:0.331, Reward: 1.0
    Exploration!
    Step 2 - Cart X: -0.035, Cart V: 0.042, Pole A: 0.034, Pole V:0.047, Reward: 1.0
    Exploitation!
    Step 3 - Cart X: -0.034, Cart V: -0.153, Pole A: 0.034, Pole V:0.350, Reward: 1.0
    Exploitation!
    Step 4 - Cart X: -0.037, Cart V: -0.349, Pole A: 0.041, Pole V:0.653, Reward: 1.0
    Exploitation!
    Step 5 - Cart X: -0.044, Cart V: -0.544, Pole A: 0.055, Pole V:0.959, Reward: 1.0
    Exploration!
    Step 6 - Cart X: -0.055, Cart V: -0.350, Pole A: 0.074, Pole V:0.684, Reward: 1.0
    Exploitation!
    Step 7 - Cart X: -0.062, Cart V: -0.546, Pole A: 0.087, Pole V:0.999, Reward: 1.0
    Exploitation!
    Step 8 - Cart X: -0.073, Cart V: -0.742, Pole A: 0.107, Pole V:1.317, Reward: 1.0
    Exploitation!
    Step 9 - Cart X: -0.088, Cart V: -0.939, Pole A: 0.134, Pole V:1.642, Reward: 1.0
    Exploitation!
    Step 10 - Cart X: -0.107, Cart V: -1.135, Pole A: 0.167, Pole V:1.973, Reward: 1.0
    Exploitation!
    Step 11 - Cart X: -0.129, Cart V: -1.331, Pole A: 0.206, Pole V:2.312, Reward: 1.0
    Exploration!
    Step 12 - Cart X: -0.156, Cart V: -1.139, Pole A: 0.252, Pole V:2.089, Reward: 1.0

