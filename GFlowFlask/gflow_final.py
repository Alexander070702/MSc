import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Initialize pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 300
SCREEN_HEIGHT = 600
BLOCK_SIZE = 30
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris with GFlowNet-inspired Agent")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
COLORS = [
    (0, 255, 255),  # Cyan
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (255, 255, 0),  # Yellow
    (0, 0, 255),    # Blue
    (255, 165, 0),  # Orange
    (160, 32, 240)  # Purple
]

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

class Piece:
    def __init__(self):
        self.x = SCREEN_WIDTH // BLOCK_SIZE // 2 - 1
        self.y = 0
        self.shape = random.choice(SHAPES)
        self.color = random.choice(COLORS)

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(SCREEN_WIDTH // BLOCK_SIZE)] for _ in range(SCREEN_HEIGHT // BLOCK_SIZE)]
    for (x, y), color in locked_positions.items():
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
    accepted_positions = [[(j, i) for j in range(SCREEN_WIDTH // BLOCK_SIZE) if grid[i][j] == BLACK] for i in range(SCREEN_HEIGHT // BLOCK_SIZE)]
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
        if y < 1:
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

# Neural Network for action prediction
class TetrisNet(nn.Module):
    def __init__(self):
        super(TetrisNet, self).__init__()
        self.fc1 = nn.Linear(10 * 20, 128)  # Input is the flattened grid (200 cells)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # Actions: left, right, rotate, down

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define a reward function
def calculate_reward(rows_cleared, current_height, holes):
    return rows_cleared * 10 - (current_height + holes * 0.5)  # Reward rows cleared, penalize height and holes

# Sample action based on network prediction
def sample_action(network, grid, device):
    tensor_grid = grid_to_tensor(grid, device)
    action_probs = torch.softmax(network(tensor_grid), dim=-1)  # Softmax for probability distribution
    action = torch.multinomial(action_probs, 1).item()  # Sample based on probabilities
    return action

# Training function for the GFlowNet-inspired agent
def train(network, optimizer, reward, state, action):
    optimizer.zero_grad()
    predicted_values = network(state)
    target = predicted_values.clone().detach()
    target[0][action] = reward
    loss = nn.MSELoss()(predicted_values, target)
    loss.backward()
    optimizer.step()

def grid_to_tensor(grid, device):
    flat_grid = np.array(grid).reshape(-1)[:200]  # Ensure exactly 200 elements
    flat_grid = flat_grid / 255  # Normalize color values to 0-1
    tensor = torch.FloatTensor(flat_grid).unsqueeze(0)  # Add batch dimension
    return tensor.to(device)

def display_action_probabilities(surface, action_probs):
    font = pygame.font.SysFont("comicsans", 20)
    actions = ["Left", "Right", "Rotate", "Down"]
    for i, prob in enumerate(action_probs.flatten()):  # Flatten array for individual access
        label = font.render(f"{actions[i]}: {prob:.2f}", 1, GREEN)
        surface.blit(label, (10, 30 + i * 20))


def main():
    net = TetrisNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    score = 0

    SCREEN.fill(BLACK)
    pygame.display.update()
    pygame.time.delay(1000)

    # Define the cooldown interval in milliseconds (e.g., 500ms)
    action_cooldown = 500
    last_action_time = pygame.time.get_ticks()  # Initialize to the current time

    while True:
        locked_positions = {}
        grid = create_grid(locked_positions)
        change_piece = False
        current_piece = Piece()
        next_piece = Piece()
        clock = pygame.time.Clock()
        fall_speed = 1.0  # Slower descent speed
        fall_time = 0
        run = True

        while run:
            grid = create_grid(locked_positions)
            fall_time += clock.get_rawtime()
            clock.tick()

            pygame.time.delay(50)  # Small delay to control game speed further

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Piece falling mechanism
            if fall_time / 1000 >= fall_speed:
                fall_time = 0
                current_piece.y += 1
                if not valid_space(current_piece, grid) and current_piece.y > 0:
                    current_piece.y -= 1
                    change_piece = True

            # Check if enough time has passed since the last action
            current_time = pygame.time.get_ticks()
            if current_time - last_action_time >= action_cooldown:
                action = sample_action(net, grid, device)
                action_probs = torch.softmax(net(grid_to_tensor(grid, device)), dim=-1).detach().cpu().numpy()
                display_action_probabilities(SCREEN, action_probs)

                # Update the last action time
                last_action_time = current_time

                # Movement logic based on action
                if action == 0:  # Move Left
                    current_piece.x -= 1
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1
                elif action == 1:  # Move Right
                    current_piece.x += 1
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1
                elif action == 2:  # Rotate
                    current_piece.rotate()
                    if not valid_space(current_piece, grid):
                        current_piece.rotate()
                elif action == 3:  # Move Down
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1
                        change_piece = True

            # Draw the piece in the grid
            shape_pos = convert_shape_format(current_piece)
            for pos in shape_pos:
                x, y = pos
                if y > -1:
                    grid[y][x] = current_piece.color

            # If we need to change the piece
            if change_piece:
                for pos in shape_pos:
                    locked_positions[(pos[0], pos[1])] = current_piece.color
                current_piece = next_piece
                next_piece = Piece()
                change_piece = False
                # Clear rows and reward system
                rows_cleared = clear_rows(grid, locked_positions)
                reward = calculate_reward(rows_cleared, current_piece.y, sum(row.count(BLACK) == 0 for row in grid))
                train(net, optimizer, reward, grid_to_tensor(grid, device), action)
                score += rows_cleared * 10

            draw_grid(SCREEN, grid)
            pygame.display.update()

            if check_lost(locked_positions):
                draw_text_middle("You Lost! Restarting...", 40, WHITE, SCREEN)
                pygame.display.update()
                pygame.time.delay(1500)
                run = False

        SCREEN.fill(BLACK)  # Clear the screen for the next game loop


if __name__ == "__main__":
    main()
