import pygame
import random

# Khởi tạo Pygame
pygame.init()

# Kích thước cửa sổ
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Mê cung ngẫu nhiên")

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_COLOR = (50, 50, 50)

# Cài đặt lưới mê cung
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Khởi tạo lưới mê cung (1 là tường, 0 là đường đi)
grid = [[1] * GRID_WIDTH for _ in range(GRID_HEIGHT)]

def generate_maze(x, y):
    grid[y][x] = 0 # Đánh dấu ô hiện tại là đường đi
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Các hướng: Đông, Tây, Nam, Bắc
    random.shuffle(directions) # Trộn ngẫu nhiên các hướng

    for dx, dy in directions:
        nx, ny = x + dx, y + dy # Tính tọa độ ô lân cận
        if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT and grid[ny][nx] == 1:
            # Nếu ô lân cận nằm trong lưới và là tường (chưa thăm)
            generate_maze(nx, ny) # Gọi đệ quy từ ô lân cận

def draw_maze():
    screen.fill(WHITE) # Xóa màn hình bằng màu trắng

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y][x] == 1: # Nếu là tường
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, BLACK, rect)
            else: # Nếu là đường đi
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, WHITE, rect) # Đường đi màu trắng (trùng màu nền)

                # Vẽ lưới (tùy chọn)
                pygame.draw.line(screen, GRID_COLOR, (x * CELL_SIZE, y * CELL_SIZE), ((x + 1) * CELL_SIZE, y * CELL_SIZE)) # Đường ngang
                pygame.draw.line(screen, GRID_COLOR, (x * CELL_SIZE, y * CELL_SIZE), (x * CELL_SIZE, (y + 1) * CELL_SIZE)) # Đường dọc


    # Vẽ viền ngoài cùng của mê cung
    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, HEIGHT), 3)

    pygame.display.flip()

# Sinh mê cung ban đầu
start_x, start_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
generate_maze(start_x, start_y)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Tạo mê cung mới
                grid = [[1] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
                start_x, start_y = random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1)
                generate_maze(start_x, start_y)

    draw_maze()

pygame.quit()