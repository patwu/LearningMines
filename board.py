import numpy as np
import copy
import Queue
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os


class Board(object):
    def __init__(self, n_height=10, n_width=10,n_mines=10):
        self.n_width = n_width
        self.n_height = n_height

        while True:
            self.n_mines = n_mines
            self.gamestate='ready'
            self.n_moves=0
            self.lastmove= None

            # random mines
            stat = [1] * self.n_mines + [0] * (n_height * n_width - self.n_mines)
            self.mines = np.asarray(np.random.permutation(stat)).reshape([n_height, n_width])

            self.show = np.asarray([[-1] * n_width] * n_height).astype(dtype=int)
            self.open_brick = 0

            self.label = np.zeros([n_height, n_width]).astype(dtype=int)
            self.mask = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]
            for row in range(n_height):
                for col in range(n_width):
                    for i in range(len(self.mask)):
                        nrow = row + self.mask[i][0]
                        ncol = col + self.mask[i][1]
                        if nrow in range(n_height) and ncol in range(n_width) and self.mines[nrow][ncol] == 1:
                            self.label[row][col] += 1

            row, col = np.random.randint(0, n_height),np.random.randint(0, n_width)
            self.play(row, col)
            if self.open_brick > 5:
                break
        self.lastmove=None
        self.n_moves=0

    def play(self, row, col):
        if self.gamestate!='ready':
            return self.gamestate,copy.deepcopy(self.show)

        self.lastmove=(row,col)
        self.n_moves+=1
        if self.show[row][col] != -1 or self.mines[row][col] == 1 or (not row in range(self.n_height)) or (not col in range(self.n_width)):
            self.gamestate='fail'
        else:
            # expand BFS
            queue = Queue.Queue()
            queue.put([row, col])
            self.open_brick += 1
            self.show[row][col] = self.label[row][col]
            while not queue.empty():
                cur = queue.get()
                row = cur[0]
                col = cur[1]
                if self.label[row][col] == 0:
                    for i in range(len(self.mask)):
                        nrow = row + self.mask[i][0]
                        ncol = col + self.mask[i][1]
                        if nrow in range(self.n_height) and ncol in range(self.n_width) and self.show[nrow][ncol] == -1 and self.mines[nrow][ncol] == 0:
                            self.show[nrow][ncol] = self.label[nrow][ncol]
                            self.open_brick += 1
                            queue.put([nrow, ncol])
            if self.open_brick + self.n_mines == self.n_width * self.n_height:
                self.gamestate='success'
        return self.gamestate,copy.deepcopy(self.show)

    def get_array(self):
        return self.show

    def get_state(self):
        return self.gamestate

    def print_board(self):
        str='+'+'--'*self.n_width+'-+\n'
        for i in range(self.n_height):
            str+='| '
            for j in range(self.n_width):
                if self.show[i][j]==-1:
                    str+='*'
                elif self.show[i][j]==0:
                    str+=' '
                else:
                    str+='%1d'%self.show[i][j]
                if self.lastmove!=None and self.lastmove==(i,j):
                    str+=')'
                else:
                    str+=' '
            str+='|\n'
        str+='+'+'--'*self.n_width+'-+\n'
        str+='state:%s\n'%self.gamestate
        str+='num mines:%d\n'%self.n_mines
        str+='num moves:%d\n'%self.n_moves
        print str

    def draw_board(self,path, policy=None):
        header = 20
        border = 10
        block = 20
        remain = 0
        n_height=self.n_height
        n_width=self.n_width
        for i in range(n_height):
            for j in range(n_width):
                if self.show[i][j] == -1:
                    remain += 1

        image_mode = 'RGB'
        image_size = (border * 2 + n_height * block, border * 2 + n_width * block + header)
        bg_color = (255, 255, 255)
        fg_color = (0, 0, 0)

        img = Image.new(image_mode, image_size, bg_color)

        draw = ImageDraw.Draw(img)
        text = 'remain: %d status: %s' %(remain,self.gamestate)
        draw.text([border, border], text, fill=fg_color)

        x = border
        if policy != None:
            mv = np.argmax(policy)
            gap=np.max(policy)-np.min(policy)
            if gap<0.001:
                gap=0.001
            minvalue=np.min(policy)
        print policy
        for i in range(n_height):
            y = border + header
            for j in range(n_width):
                if policy == None:
                    draw.rectangle([(x, y), (x + block, y + block)], fill=(255, 255, 255), outline=fg_color)
                else:
                    norm = int((policy[i][j]-minvalue)/gap * 255)
                    fill = (255 - norm, 255, 255)
                    if i * n_width + j == mv:
                        fill = (fill[0], 128, fill[2])
                    draw.rectangle([(x, y), (x + block, y + block)], fill=fill, outline=fg_color)
                if self.show[i][j] != -1:
                    draw.text([x + block / 4, y + block / 4], str(self.show[i][j]), fill=fg_color)
                y += block
            x += block
        filename=os.path.join(path,'%s.png'%self.n_moves)
        f = open(filename, 'w')
        img.save(f)
        f.close()


def main():
    board = Board(10,10)
    board.print_board()
    board.draw_board('tmp')
    board.play(5, 5)
    board.print_board()
    board.draw_board('tmp')

if __name__ == '__main__':
    main()
