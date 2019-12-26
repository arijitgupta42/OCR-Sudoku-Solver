import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import time

def solve(board):
    find = find_null(board)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(board, i, (row, col)):
            board[row][col] = i

            if solve(board):
                return True

            board[row][col] = 0

    return False


def valid(board, num, pos):
    # Check row
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    # Check boardx
    boardx_x = pos[1] // 3
    boardx_y = pos[0] // 3

    for i in range(boardx_y*3, boardx_y*3 + 3):
        for j in range(boardx_x * 3, boardx_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True


def show_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")

        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")

            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")


def find_null(board):
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 0:
                return (i, j)  # row, col

    return None

#open the image
img = Image.open('images/sudoku.png').convert('LA')
#resizing image so that each digit is comaparable in size to the MNIST digits
img = img.resize((280,280), Image.ANTIALIAS)
#take only the brightness value from each pixel of the image
array = np.array(img)[:,:,0]
#invert the image (this is how MNIST digits is formatted)
array = 255-array
#this will be the width and length of each sub-image
divisor = array.shape[0]//9

puzzle = []
for i in range(9):
    row = []
    for j in range(9):
        #slice image, reshape it to 28x28 (mnist reader size)
        row.append(cv2.resize(array[i*divisor:(i+1)*divisor,
                                    j*divisor:(j+1)*divisor][3:-3, 3:-3], #the 3:-3 slice removes the boardrders from each image
                              dsize=(28,28), 
                              interpolation=cv2.INTER_CUBIC))
    puzzle.append(row)

#load the trained model
model = tf.keras.models.load_model('./models/my_model.h5')
print(model.summary())
#Restore the weights
model.load_weights('./checkpoints/my_checkpoint')
#create a 9x9 array of 0s (the sudoku solver doesn't use numpy so I won't here)
template = [
    [0 for _ in range(9)] for _ in range(9)
]

for i, row in enumerate(puzzle):
    for j, image in enumerate(row):
        #if the brightness is above 6, then use the model
        if np.mean(image) > 6:
            #this line of code sets the puzzle's value to the model's prediction
            #the preprocessing happens inside the predict call
            template[i][j] = model.predict_classes(image.reshape(1,28,28,1) \
                                                   .astype('float32')/255)[0]
            


show_board(template)
start =time.time()
if solve(template):
    print("_______________________")
    print("_______________________")
    print("")
    end = time.time()
    show_board(template)
    print("Solved in {0:9.5f}s".format(end-start))
else:
    print("Cannot be solved")
