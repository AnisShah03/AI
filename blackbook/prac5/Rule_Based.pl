man(lou).
man(pete).
man(ian).
man(peter).

woman(pauline).
woman(catchy).
woman(lucy).

parent(ian,lucy).
parent(ian,peter).
parent(cathy,ian).
parent(pete,ian).
parent(lou,pete).
parent(lou,pauline).

mother(X,Y):-woman(X),parent(X,Y),(X\=Y).
father(X,Y):-man(X),parent(X,Y),(X\=Y).

sibling(X,Y):-parent(Z,X),parent(Z,Y),(X\=Y).
brother(X,Y):-man(X),sibling(X,Y),(X\=Y).
sister(X,Y):-woman(X),sibling(X,Y),(X\=Y).

grandfather(X,Y):-father(X,Z),parent(Z,Y),(X\=Y).
grandmother(X,Y):-mother(X,Y),parent(Z,Y),(X\=Y).

ancestor(X,Y):-parent(X,Y),(X\=Y).
ancestor(X,Y):- parent(X,Z),ancestor(Z,Y),(X\=Y).

