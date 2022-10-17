#include "variables.h"

Point pt[Np];
Line wall[Nw];
vec door;
double c; // noise ratio
double len = 2*_r + 5*paraB;
int number;
list<int> grids[Ng_x][Ng_y];
double save_position[Np][2][2000];
double save_velocity[Np][2][2000];
double save_force[Np][8][2000];
int save_flag[2000][Np];
int Out_number[2000];
int ET[Np];
double theta = 90;

double _left = -5.0;
double _right = 5.0;
double _up = 5.0;
double _down = -5.0;
double width = 1.0;


