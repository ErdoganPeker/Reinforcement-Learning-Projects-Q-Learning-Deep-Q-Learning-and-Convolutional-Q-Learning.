import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Conv2D
from collections import deque


NBACTIONS = 3
IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

OBSERVEPERIOD = 2000
GAMMA = 0.975
BATCH_SIZE = 64

ExpReplay_CAPACITY = 2000


class Agent:
    
    def __init__(self):
        self.model = self.createModel()
        self.ExpReplay = deque()
        self.steps = 0
        self.epsilon = 1.0
    
    def createModel(self):
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=4, strides = (2,2), input_shape = (IMGHEIGHT,IMGWIDTH,IMGHISTORY),padding = "same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64,kernel_size=4,strides=(2,2),padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64,kernel_size=3,strides=(1,1),padding="same"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dense(units= NBACTIONS, activation="linear"))
        
        model.compile(loss = "mse", optimizer="adam")
        
        return model
    
    def FindBestAct(self, s):
        if random.random() < self.epsilon or self.steps < OBSERVEPERIOD:
            return random.randint(0,NBACTIONS - 1)
        else:
            qvalue = self.model.predict(s)
            bestA = np.argmax(qvalue)
            return bestA
    
    def CaptureSample(self, sample):
        self.ExpReplay.append(sample)
        if len(self.ExpReplay) > ExpReplay_CAPACITY:
            self.ExpReplay.popleft()
        
        self.steps += 1 
        
        self.epsilon = 1.0
        if self.steps > OBSERVEPERIOD:
            self.epsilon = 0.75
            if self.steps > 7000:
                self.epsilon = 0.5
            if self.steps > 14000:
                self.epsilon = 0.25
            if self.steps > 30000:
                self.epsilon = 0.15
            if self.steps > 45000:
                self.epsilon = 0.1
            if self.steps > 70000:
                self.epsilon = 0.05
    
    def Process(self):
        if self.steps > OBSERVEPERIOD:
            minibatch = random.sample(self.ExpReplay, BATCH_SIZE)
            batchlen = len(minibatch)
            
            inputs = np.zeros((BATCH_SIZE,IMGHEIGHT,IMGWIDTH,IMGHISTORY))
            targets = np.zeros((inputs.shape[0],NBACTIONS))
            
            Q_sa = 0
            
            for i in range(batchlen):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                
                inputs[i:i + 1] = state_t
                targets[i]  = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1)
                
                if state_t1 is None:
                    targets[i,action_t] = reward_t
                else:
                    targets[i,action_t] = reward_t + GAMMA*np.max(Q_sa)
                
            
            self.model.fit(inputs, targets ,batch_size= BATCH_SIZE, epochs=1, verbose=0)
    
import pygame
import random

FPS = 60

WINDOW_WIDTH = 400
WINDOW_HEIGHT = 420
GAME_HEIGHT = 400

PADDLE_WIDTH = 15
PADDLE_HEIGHT = 60
PADDLE_BUFFER = 15

BALL_WIDTH = 20
BALL_HEIGHT = 20

PADDLE_SPEED = 3
BALL_X_SPEED = 2
BALL_Y_SPEED = 2

WHITE = (255,255,255)
BLACK = (0,0,0)

screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))

def drawPaddle(switch, paddleYPos):
    
    if switch == "left":
        paddle = pygame.Rect(PADDLE_BUFFER, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
    elif switch == "right":
        paddle = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddleYPos, PADDLE_WIDTH, PADDLE_HEIGHT)
        
    pygame.draw.rect(screen, WHITE, paddle)
 
def drawBall(ballXPos, ballYPos):
    
    ball = pygame.Rect(ballXPos, ballYPos, BALL_WIDTH, BALL_HEIGHT)
    
    pygame.draw.rect(screen, WHITE, ball)

def updatePaddle(switch, action, paddleYPos, ballYPos):
    dft = 7.5 
    
    # AGENT
    if switch == "left":
        if action == 1:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft
        if action == 2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
            
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT
    elif switch == "right":
        if paddleYPos + PADDLE_HEIGHT/2 < ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos + PADDLE_SPEED*dft
        if paddleYPos + PADDLE_HEIGHT/2 > ballYPos + BALL_HEIGHT/2:
            paddleYPos = paddleYPos - PADDLE_SPEED*dft   
            
        if paddleYPos < 0:
            paddleYPos = 0
        if paddleYPos > GAME_HEIGHT - PADDLE_HEIGHT:
            paddleYPos = GAME_HEIGHT - PADDLE_HEIGHT
    
    return paddleYPos

def updateBall(paddle1YPos, paddle2YPos, ballXPos, ballYPos, ballXDirection, ballYDirection,DeltaFrameTime):
    
    dft = 7.5
    
    ballXPos = ballXPos + ballXDirection*BALL_X_SPEED*dft
    ballYPos = ballYPos + ballYDirection*BALL_Y_SPEED*dft
    
    score = -0.05
    
    # agent
    if (ballXPos <= (PADDLE_BUFFER + PADDLE_WIDTH)) and ((ballYPos + BALL_HEIGHT) >= paddle1YPos) and (ballYPos <= (paddle1YPos + PADDLE_HEIGHT)) and (ballXDirection == -1):
        
        ballXDirection = 1 
        
        score = 10
        
    elif (ballXPos <= 0):
        
        ballXDirection = 1
        
        score = -10 
        
        return [score, ballXPos ,ballYPos ,ballXDirection, ballYDirection]
    
    if ((ballXPos >= (WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER)) and ((ballYPos + BALL_HEIGHT)>= paddle2YPos) and (ballYPos <= (paddle2YPos + PADDLE_HEIGHT)) and (ballXDirection == 1)):
        
        ballXDirection = -1
    
    elif(ballXPos >= WINDOW_WIDTH - BALL_WIDTH):
        
        ballXDirection = -1
        
        return [score, ballXPos,ballYPos, ballXDirection, ballYDirection]
    
    if ballYPos <= 0:
        
        ballYPos = 0
        
        ballYDirection = 1
        
    elif ballYPos >= GAME_HEIGHT - BALL_HEIGHT:
        
        ballYPos = GAME_HEIGHT - BALL_HEIGHT
        
        ballYDirection = -1
        
    return [score, ballXPos,ballYPos,ballXDirection,ballYDirection]
    
    
    
class PongGame:
    
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Pong DCQL Env")
        
        self.paddle1YPos = GAME_HEIGHT/2 - PADDLE_HEIGHT/2
        self.paddle2YPos = GAME_HEIGHT/2 - PADDLE_HEIGHT/2
        
        self.ballXPos = WINDOW_WIDTH/2
        
        self.clock = pygame.time.Clock()
        
        self.GScore = 0.0
        
        self.ballXDirection = random.sample([-1,1],1)[0]
        self.ballYDirection = random.sample([-1,1],1)[0]
        
        self.ballYPos = random.randint(0,9)*(WINDOW_HEIGHT - BALL_HEIGHT)/9
        
        
    def InitialDisplay(self):
        
        pygame.event.pump()
        
        screen.fill(BLACK)
        
        drawPaddle("left", self.paddle1YPos)
        drawPaddle("right",self.paddle2YPos)
        
        drawBall(self.ballXPos, self.ballYPos)
        
        pygame.display.flip()
    
    def PlayNextMove(self, action):
        
        DeltaFrameTime = self.clock.tick(FPS)
        
        pygame.event.pump()
        
        score = 0
        
        screen.fill(BLACK)
        
        self.paddle1YPos = updatePaddle("left", action, self.paddle1YPos, self.ballYPos)
        drawPaddle("left", self.paddle1YPos)

        self.paddle2YPos = updatePaddle("right", action, self.paddle2YPos, self.ballYPos)
        drawPaddle("right", self.paddle2YPos)
        
        [score, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection] = updateBall(self.paddle1YPos, self.paddle2YPos, self.ballXPos, self.ballYPos, self.ballXDirection, self.ballYDirection,DeltaFrameTime)
        
        drawBall(self.ballXPos, self.ballYPos)
        
        if ( score > 0.5 or score < -0.5):
            self.GScore = self.GScore*0.9 + 0.1*score 
            
        ScreenImage = pygame.surfarray.array3d(pygame.display.get_surface())
        
        pygame.display.flip()
        
        return [score, ScreenImage]
        


import numpy as np
import skimage as skimage
import warnings
warnings.filterwarnings("ignore")

TOTAL_TrainTime = 100000

IMGHEIGHT = 40
IMGWIDTH = 40
IMGHISTORY = 4

def ProcessGameImage(RawImage):
    
    GreyImage = skimage.color.rgb2gray(RawImage)
    
    CroppedImage = GreyImage[0:400,0:400]
    
    ReducedImage = skimage.transform.resize(CroppedImage,(IMGHEIGHT,IMGWIDTH))
    
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage, out_range = (0,255))
    
    ReducedImage = ReducedImage / 128
    
    return ReducedImage
        
def TrainExperiment():
    
    TrainHistory = []
    
    TheGame = PongGame()
    
    TheGame.InitialDisplay()
    
    TheAgent = Agent()
    
    BestAction = 0
    
    [InitialScore, InitialScreenImage] = TheGame.PlayNextMove(BestAction)
    InitialGameImage = ProcessGameImage(InitialScreenImage)
    
    GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis = 2)
    
    GameState = GameState.reshape(1, GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    
    for i in range(TOTAL_TrainTime):
        
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore, NewScreenImage] = TheGame.PlayNextMove(BestAction)
        
        NewGameImage = ProcessGameImage(NewScreenImage)
        
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage, GameState[:,:,:,:3], axis = 3)
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 250 == 0:
            print("Train time: ",i, " game score: ",TheGame.GScore)
            TrainHistory.append(TheGame.GScore)
            
        
TrainExperiment()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




































    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    