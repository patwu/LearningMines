import board

b=board.Board()

while b.get_state()=='ready':
    b.print_board()
    print 'input row'
    row=input()
    print 'input col'
    col=input()
    b.play(row,col)

b.print_board()
