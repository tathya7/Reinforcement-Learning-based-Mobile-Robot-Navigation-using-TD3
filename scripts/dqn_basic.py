import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt


# Define the environment dimensions
state_size = 100  # Example state size
action_size = 20  # Example number of actions

# Hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount rate
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995

# Create a Deep Q-learning Network model
def build_model(state_size, action_size):
    model = models.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=learning_rate))
    return model

# Instantiate the DQN model
model = build_model(state_size, action_size)
def check_pos(x, y):
    # Check if the random goal position is located on an obstacle and do not accept it if it is
    goal_ok = False
    
    if x > -0.55 and 1.7 > y > -1.7:
        goal_ok = True

    return goal_ok
# Simulated environment interaction (placeholders)
def get_state():
    # Placeholder for state retrieval logic
    return np.random.rand(state_size)

def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.randint(action_size)
    act_values = model.predict(state.reshape(1, -1))
    return np.argmax(act_values[0])

def take_action(action):
    # Placeholder for action logic in the environment
    # Return next state, reward, done status
    next_state = np.random.rand(state_size)
    reward = np.random.rand()
    done = np.random.choice([True, False], p=[0.1, 0.9])
    return next_state, reward, done

# Plotting and training variables
average_q_values = []
max_q_values = []
iterations = []

# Training loop
def train_model():
    global epsilon
    state = get_state()
    for iteration in range(200):  # Number of training iterations
        action = choose_action(state)
        next_state, reward, done = take_action(action)
        target = reward
        if not done:
            target = (reward + gamma * np.max(model.predict(next_state.reshape(1, -1))[0]))
        target_f = model.predict(state.reshape(1, -1))
        target_f[0][action] = target
        model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        state = next_state
        if done:
            state = get_state()
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Store Q-values for plotting
        current_q_values = target_f[0]
        average_q_values.append(np.mean(current_q_values))
        max_q_values.append(np.max(current_q_values))
        iterations.append(iteration)

# Run training
train_model()

# Plot average Q-values
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(iterations, average_q_values, label='Average Q-value', color='blue')
plt.title('Average Q-values over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Average Q-value')
plt.grid(True)

# Plot max Q-values
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(iterations, max_q_values, label='Max Q-value', color='green')
plt.title('Max Q-values over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Max Q-value')
plt.grid(True)

plt.tight_layout()