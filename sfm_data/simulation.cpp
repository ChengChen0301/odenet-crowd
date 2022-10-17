#include<cstring>

#include "simulation.h"
#include "variables.h"

using namespace std;

// generate non-overlap points
void initialize(){
    /*------------------set walls and door-----------------*/
    wall[0].set_X(_right, _up, _left, _up);
    wall[1].set_X(_left, _up, _left, _down);
    wall[2].set_X(_left, _down, _right, _down);
    wall[3].set_X(_right, _down, _right, -width);
    wall[4].set_X(_right, width, _right, _up);
    door.set_X(_right,0.0);
    /*------------------set walls and door-----------------*/


    /*------------------set point position-----------------*/
    number = Np; // initial number of points
    vec temp;
    int idx, idy;
    double _left_ = -5.0;
    double _down_ = -5.0;
    double _right_ = 5.0;
    double _up_ = 5.0;
    double given[Np][2] = {2.57867, -4.47536, 4.53889, -4.52194, 3.43042, -4.65334, 1.95727, -4.6807, 1.31251, -4.44965};

    double x = (double)rand() / RAND_MAX *(_right_ - _left_ - 2.0*_r) + _left_ + _r; // the first particle
    double y = (double)rand() / RAND_MAX *(_up_ - _down_ - 2.0*_r) + _down_ + _r;
    // x = given[0][0];
    // y = given[0][1];
    idx = (int)((x - _left)/len);
    idy = (int)((y - _down)/len);
    temp.set_X(x,y); pt[0].set_C(temp);
    pt[0].set_G(idx, idy);
    grids[idx][idy].push_back(0); // put the label "0" into the corresponding grid
    pt[0].set_flag(1); // in the room
    // pt[0].set_weight((double)rand() / RAND_MAX * (100.0 - 60.0) + 60.0);

    for(int i=1; i<Np; i++){ // rest particles
        if(i % 2 == 0)
            pt[i].set_weight(80.0);
        int flag = 1;
        while(flag == 1){
            x = (double)rand() / RAND_MAX *(_right_ - _left_ - 2.0*_r) + _left_ + _r;
            y = (double)rand() / RAND_MAX *(_up_ - _down_ - 2.0*_r) + _down_ + _r;
            // x = given[i][0];
            // y = given[i][1];
            temp.set_X(x,y);
            int index = 0;
            for (int j=0; j<i; j++){
                if(temp.distance_wrt_point(pt[j].get_C()) <= 2.0*_r){
                    index = j;
                    break;
                }
                index = j+1;
            }
            if(index >= i && temp.distance_wrt_point(pt[i-1].get_C()) > 2.0*_r){
                flag = 0;
                pt[i].set_C(temp);
                idx = (int)((x - _left)/len);
                idy = (int)((y - _down)/len);
                pt[i].set_G(idx,idy);
                grids[idx][idy].push_back(i); //put the label "i" into the corresponding grid
                pt[i].set_flag(1);
                // pt[i].set_weight((double)rand() / RAND_MAX * (100.0 - 60.0) + 60.0);
            }
        }
    }
    /*------------------set point position-----------------*/
};


double generateGaussianNoise(double mu, double sigma){
    const double epsilon = numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;

    static double z0, z1;
    static bool generate;
    generate = !generate;

    if(!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do{
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    }
    while ( u1 <= epsilon );
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
};


void computeForce(){
    vec targetV, force, dir, noise, temp;
    double dist, touch, u1, u2;
    int idx, idy, iloc, jloc; // new grid index and old index
    int flag = 1; // used in generating noise

    //calculate the force for each ball
    // #pragma omp parallel num_threads(4)
    // #pragma omp for
    for(int i=0; i<Np; i++){
        if(pt[i].get_flag() == 1){ // still in the room
            force.set_X(0.0,0.0);

            iloc = pt[i].get_G()[0];
            jloc = pt[i].get_G()[1]; // find its grid
            // go through the adjacent 9 grids
            for(int j=-1; j<2; j++){
                for(int k=-1; k<2; k++){
                    // compute points whose labels are in the grids
                    if(iloc+j>=0 && jloc+k>=0 && (!grids[iloc+j][jloc+k].empty())){
                        for(list<int>::iterator iter=grids[iloc+j][jloc+k].begin(); iter!=grids[iloc+j][jloc+k].end();iter++){
                            int label = *iter;
                            if(label!=i && pt[label].get_flag()==1){
                                dist = pt[i].get_C().distance_wrt_point(pt[label].get_C());
                                touch = pt[i].get_C().touch_point(pt[label].get_C());
                                dir = pt[i].get_C().direction_from_point(pt[label].get_C());
                                force = force + dir*(paraA*exp((2.0*_r - dist)/paraB)) + dir*(k_bump*touch)\
                                        + dir.normalvector()*(K_friction*touch*(pt[label].get_V()-pt[i].get_V()).inner_with(dir.normalvector()));
                            }
                        }
                    }
                }
            }
            pt[i].set_F3(force);

            for(int j=0; j<Nw; j++){ //force between ball and walls
                dist = pt[i].get_C().distance_wrt_line(wall[j]);
                touch = pt[i].get_C().touch_line(wall[j]);
                dir = pt[i].get_C().direction_from_line(wall[j]);
                force = force + dir*(paraA*exp((_r - dist)/paraB)) + dir*(k_bump*touch)\
                        - dir.normalvector()*(K_friction*touch*pt[i].get_V().inner_with(dir.normalvector()));
            }
            pt[i].set_F1(force - pt[i].get_F3());

            targetV = door.direction_from_point(pt[i].get_C())*v;

            while(flag == 1){
                u1 = generateGaussianNoise(0,1);
                u2 = generateGaussianNoise(0,1);
                if(abs(u1)<=2 && abs(u2)<=2){ // 2-sigma cut-off
                    noise.set_X(u1, u2);
                    flag = 0;
                }
            }

            pt[i].set_F2((targetV - pt[i].get_V())/tau*pt[i].get_weight());
            pt[i].set_F(force + (targetV - pt[i].get_V())/tau*pt[i].get_weight());
        }
        else{
            force.set_X(0.0,0.0);
            pt[i].set_F(force);
            pt[i].set_F1(force);
            pt[i].set_F2(force);
            pt[i].set_F3(force);
        }
    }
};


void update(int t){
    int idx, idy, iloc, jloc;
    for(int i=0; i<Np; i++){
        if(pt[i].get_flag() == 1){ // still in the room
            iloc = pt[i].get_G()[0];
            jloc = pt[i].get_G()[1];

            pt[i].set_A(pt[i].get_F()/pt[i].get_weight()); //update acceleration
            pt[i].set_V(pt[i].get_V() + pt[i].get_A()*delta); // update speed
            pt[i].set_C(pt[i].get_C() + pt[i].get_V()*delta); // update position

            if(pt[i].get_C().get_x() >= door.get_x()){ // update flag
                pt[i].set_flag(0);
                number--;
                ET[i] = t;
            }

            idx = (int)((pt[i].get_C().get_x() - _left)/len);
            idy = (int)((pt[i].get_C().get_y() - _down)/len);

            if(idx != iloc || idy != jloc){ // if grid information changes

                for(list<int>::iterator iter=grids[iloc][jloc].begin(); iter!=grids[iloc][jloc].end();){
                    if(*iter == i){
                        iter = grids[iloc][jloc].erase(iter); //remove it from the old grid
                        continue;
                    }
                    iter++;
                } // delete the point from the old grid
                if(pt[i].get_flag() == 1){
                    grids[idx][idy].push_back(i); // add the point into the new grid
                    pt[i].set_G(idx,idy); // update grid
                }
            }
        }
      
    }
};

void savedata(int arg){
    // arg = arg/100;
    for(int i=0; i<Np; i++){
        save_position[i][0][arg] = pt[i].get_C().get_x();
        save_position[i][1][arg] = pt[i].get_C().get_y();
        save_flag[arg][i] = pt[i].get_flag();
        save_velocity[i][0][arg] = pt[i].get_V().get_x();
        save_velocity[i][1][arg] = pt[i].get_V().get_y();
        save_force[i][0][arg] = pt[i].get_F1().get_x();
        save_force[i][1][arg] = pt[i].get_F1().get_y();
        save_force[i][2][arg] = pt[i].get_F2().get_x();
        save_force[i][3][arg] = pt[i].get_F2().get_y();
        save_force[i][4][arg] = pt[i].get_F3().get_x();
        save_force[i][5][arg] = pt[i].get_F3().get_y();
        save_force[i][6][arg] = pt[i].get_F().get_x();
        save_force[i][7][arg] = pt[i].get_F().get_y();
        Out_number[arg] = Np - number;
    }
};
