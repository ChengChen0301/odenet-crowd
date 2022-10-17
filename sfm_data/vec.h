#ifndef SIMULATION_VEC_H
#define SIMULATION_VEC_H

#include <cmath>

#include "Line.h"

using namespace std;

class vec{
private:
    double x_component;
    double y_component;
public:

    vec()
    {
        x_component = 0.0;
        y_component = 0.0;
    };

    void set_x(double arg){
        x_component = arg;
    };
    void set_y(double arg){
        y_component = arg;
    };
    void set_X(double arg1, double arg2){
        x_component = arg1;
        y_component = arg2;
    };

    ///////////////////////
    ///visit member data///
    ///////////////////////
    double get_x();
    double get_y();
    vec get_X();

    /////////////////
    ///operations////
    /////////////////
    vec operator + (vec arg);
    vec operator - (vec arg);
    vec operator / (double arg);
    vec operator * (double arg); // the expression is X*arg. not arg*X

    double inner_with(vec other); // return inner product with another vector
    double norm(); //return length of a vector
    double distance_wrt_point(vec other);
    vec footpoint_wrt(Line arg); // return foot point in a line
    double distance_wrt_line(Line arg);
    vec direction_from_point(vec other); // pointing from the other to itself
    vec direction_from_line(Line arg); // vertically pointing to itself
    vec normalvector(); // return its normal vector
    double touch_point(vec other); // the g(x) function in the formula
    double touch_line(Line arg);
};
#endif //SIMULATION_VEC_H
