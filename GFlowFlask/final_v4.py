import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 800
BLOCK_SIZE = 30
INFO_PANEL_WIDTH = 240  # Increased width for better text display
SCREEN = pygame.display.set_mode((SCREEN_WIDTH + INFO_PANEL_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris with Enhanced GFlowNet-inspired Agent")

# Dynamic grid size based on screen settings
GRID_COLUMNS = SCREEN_WIDTH // BLOCK_SIZE
GRID_ROWS = SCREEN_HEIGHT // BLOCK_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)

# Tetris shapes
SHAPES = [
    [[1, 1, 1, 1]],  # I-shape
    [[1, 1, 1], [0, 1, 0]],  # T-shape
    [[1, 1], [1, 1]],  # O-shape
    [[1, 1, 0], [0, 1, 1]],  # Z-shape
    [[0, 1, 1], [1, 1, 0]],  # S-shape
    [[1, 1, 1], [1, 0, 0]],  # L-shape
    [[1, 1, 1], [0, 0, 1]]   # J-shape
]

# Fonts
font_small = pygame.font.SysFont("comicsans", 18)
font_medium = pygame.font.SysFont("comicsans", 24, bold=True)

class Piece:
    def __init__(self):
        self.x = GRID_COLUMNS // 2 - 1
        self.y = 0
        self.shape = random.choice(SHAPES)
        self.color = random.choice([(0, 255, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 165, 0), (160, 32, 240)])

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(GRID_COLUMNS)] for _ in range(GRID_ROWS)]
    for (x, y), color in locked_positions.items():
        if y < GRID_ROWS:
            grid[y][x] = color
    return grid

def convert_shape_format(piece):
    positions = []
    shape = piece.shape
    for i, line in enumerate(shape):
        for j, column in enumerate(line):
            if column == 1:
                positions.append((piece.x + j, piece.y + i))
    return positions

def valid_space(piece, grid):
    accepted_positions = [[(j, i) for j in range(GRID_COLUMNS) if grid[i][j] == BLACK] for i in range(GRID_ROWS)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(piece)
    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] >= 0:
                return False
    return True

def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y == 0:
            return True
    return False

def clear_rows(grid, locked):
    increment = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if BLACK not in row:
            increment += 1
            index = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if increment > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < index:
                newKey = (x, y + increment)
                locked[newKey] = locked.pop(key)
    return increment

def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont("comicsans", size, bold=True)
    label = font.render(text, 1, color)
    surface.blit(label, (SCREEN_WIDTH // 2 - label.get_width() // 2, SCREEN_HEIGHT // 2 - label.get_height() // 2))

def draw_grid(surface, grid):
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j], (j*BLOCK_SIZE, i*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    pygame.draw.rect(surface, WHITE, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 4)

class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        self.fc1 = nn.Linear(GRID_COLUMNS * GRID_ROWS, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FlowManager:
    def __init__(self):
        self.flows = torch.ones(4)  # Start with equal flow for each action

    def sample_action(self):
        action_probs = torch.softmax(self.flows, dim=0)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs

    def update_flows(self, action, reward, learning_rate=0.05, decay=0.05, dominant_decay_threshold=2.5):
        # Apply decay to all actions
        self.flows = self.flows * (1 - decay)
        
        # Apply stronger decay if the chosen action becomes overly dominant
        if self.flows[action] > dominant_decay_threshold:
            self.flows[action] *= 0.8  # Stronger decay for the overly dominant action

        # Scale reward for smoother updates
        scaled_reward = min(10, reward) / 10.0  # Normalize reward to prevent high dominance
        self.flows[action] += learning_rate * scaled_reward

def calculate_reward(cumulative_rows_cleared, cumulative_height_increase, cumulative_holes, moves_since_last_clear, last_action, repeated_action_count):
    # Encourage clearing rows and maintaining low height with fewer holes
    row_reward = 300 * cumulative_rows_cleared if cumulative_rows_cleared > 0 else 10
    height_reward = max(5, 100 - cumulative_height_increase)  # Reduced base for height reward
    hole_penalty = max(5, 30 - (cumulative_holes * 2))  # Reduced hole penalty for smoother balancing
    sequence_penalty = max(1, 10 - moves_since_last_clear)  # Reduced sequence penalty

    # Diversity penalty for excessive repeated "down" actions without clearing rows
    diversity_penalty = -20 if last_action == 3 and repeated_action_count > 5 else 0

    # Total reward calculation
    reward = row_reward + height_reward + hole_penalty + sequence_penalty + diversity_penalty
    reward = max(5, reward)  # Minimum reward to ensure positivity

    return reward



class ReplayBuffer:
    def __init__(self, capacity=500):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience, priority=None):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        batch = [self.buffer[i] for i in indices]
        return batch

def train_from_replay(net, optimizer, buffer, batch_size=32):
    if len(buffer) < batch_size:
        return
    batch = buffer.sample(batch_size)
    for state, action, reward in batch:
        optimizer.zero_grad()
        predicted_values = net(state)
        target = predicted_values.clone().detach()
        target[0][action] = reward
        loss = nn.MSELoss()(predicted_values, target)
        loss.backward()
        optimizer.step()

def train(net, optimizer, reward, state, action):
    optimizer.zero_grad()
    predicted_values = net(state)
    target = predicted_values.clone().detach()
    target[0][action] = reward
    loss = nn.MSELoss()(predicted_values, target)
    loss.backward()
    optimizer.step()

def grid_to_tensor(grid, device):
    flat_grid = np.array(grid).reshape(-1)[:GRID_COLUMNS * GRID_ROWS]
    flat_grid = flat_grid / 255
    tensor = torch.FloatTensor(flat_grid).unsqueeze(0)
    return tensor.to(device)

def display_info_panel(surface, score, reward, action_probs, flows):
    surface.fill(BLACK, (SCREEN_WIDTH, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT))
    
    score_label = font_medium.render(f"Score: {score}", 1, WHITE)
    reward_label = font_small.render(f"Reward: {reward:.2f}", 1, CYAN)
    surface.blit(score_label, (SCREEN_WIDTH + 10, 10))
    surface.blit(reward_label, (SCREEN_WIDTH + 10, 50))
    
    info_text = [
        "Agent Behavior",
        "1. Actions sampled based on probabilities.",
        "2. Rewards guide learning.",
        "3. Rows cleared give rewards!",
        "4. Height & holes penalized."
    ]
    for i, line in enumerate(info_text):
        info_label = font_small.render(line, 1, WHITE)
        surface.blit(info_label, (SCREEN_WIDTH + 10, 90 + i * 25))
    
    actions = ["Left", "Right", "Rotate", "Down"]
    for i, (prob, flow) in enumerate(zip(action_probs, flows)):
        label = font_small.render(f"{actions[i]}: {prob:.2f} (Flow: {flow:.2f})", 1, GREEN)
        surface.blit(label, (SCREEN_WIDTH + 10, 230 + i * 25))


def adaptive_epsilon(action_probs, min_epsilon=0.1, max_epsilon=0.5):
    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
    return max(min_epsilon, min(max_epsilon, entropy / torch.log(torch.tensor(len(action_probs)))))

def main():
    net = TetrisNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    score = 0
    SCREEN.fill(BLACK)
    pygame.display.update()
    pygame.time.delay(1000)

    flow_manager = FlowManager()
    replay_buffer = ReplayBuffer()
    last_action_time = pygame.time.get_ticks()
    action_cooldown = 200
    reward = 0.0

    # Track sequence-based variables
    moves_since_last_clear = 0
    cumulative_height_increase = 0
    cumulative_holes = 0
    cumulative_rows_cleared = 0
    last_action = None
    repeated_action_count = 0

    while True:
        locked_positions = {}
        grid = create_grid(locked_positions)
        change_piece = False
        current_piece = Piece()
        next_piece = Piece()
        clock = pygame.time.Clock()
        fall_speed = 0.5
        fall_time = 0
        run = True

        action_probs = torch.softmax(flow_manager.flows, dim=0)

        while run:
            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            clock.tick()
            pygame.time.delay(50)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            if fall_time / 1000 >= fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not valid_space(current_piece, grid) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            current_time = pygame.time.get_ticks()
            if current_time - last_action_time >= action_cooldown:
                action, action_probs = flow_manager.sample_action()
                epsilon = adaptive_epsilon(action_probs)
                if random.random() < epsilon:
                    action = random.randint(0, 3)
                last_action_time = current_time

                # Check if the action is repeated
                if action == last_action:
                    repeated_action_count += 1
                else:
                    repeated_action_count = 0
                last_action = action

                # Execute the chosen action
                if action == 0:
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1
                elif action == 1:
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                elif action == 2:
                    current_piece.rotate()
                    if not valid_space(current_piece, grid):
                        current_piece.rotate()
                elif action == 3:
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1
                        change_piece = True

            shape_pos = convert_shape_format(current_piece)
            for pos in shape_pos:
                x, y = pos
                if y > -1:
                    grid[y][x] = current_piece.color

            if change_piece:
                for pos in shape_pos:
                    locked_positions[(pos[0], pos[1])] = current_piece.color
                current_piece = next_piece
                next_piece = Piece()
                change_piece = False
                rows_cleared = clear_rows(grid, locked_positions)

                # Track cumulative metrics for reward calculation
                cumulative_rows_cleared += rows_cleared
                cumulative_height_increase += current_piece.y
                cumulative_holes += sum(1 for row in grid for cell in row if cell == BLACK)
                moves_since_last_clear += 1 if rows_cleared == 0 else 0

                # Calculate reward based on cumulative metrics
                reward = calculate_reward(
                    cumulative_rows_cleared,
                    cumulative_height_increase,
                    cumulative_holes,
                    moves_since_last_clear,
                    last_action,
                    repeated_action_count
                )

                # Reset cumulative metrics after issuing reward
                if rows_cleared > 0:
                    moves_since_last_clear = 0
                    cumulative_rows_cleared = 0
                    cumulative_height_increase = 0
                    cumulative_holes = 0

                flow_manager.update_flows(action, reward)
                replay_buffer.add((grid_to_tensor(grid, device), action, reward))
                score += reward

                if len(replay_buffer.buffer) >= 10:
                    batch = replay_buffer.sample(10)
                    for state, action, reward in batch:
                        train(net, optimizer, reward, state, action)

            draw_grid(SCREEN, grid)
            display_info_panel(SCREEN, score, reward, action_probs, flow_manager.flows)
            pygame.display.update()

            if check_lost(locked_positions):
                draw_text_middle("You Lost! Restarting...", 40, WHITE, SCREEN)
                pygame.display.update()
                pygame.time.delay(1500)
                run = False

        SCREEN.fill(BLACK)

if __name__ == "__main__":
    main()
