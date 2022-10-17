#include <cfloat>

#include "vec.h"
#include "variables.h"

double vec::get_x(){
    return x_component;
};
double vec::get_y(){
    return y_component;
};
vec vec::get_X(){
    vec temp;
    temp.set_X(x_component,y_component);
    return temp;
};


vec vec::operator + (vec arg){
    vec temp;
    temp.x_component = x_component + arg.x_component;
    temp.y_component = y_component + arg.y_component;
    return temp;
};
vec vec::operator - (vec arg){
    vec temp;
    temp.x_component = x_component - arg.x_component;
    temp.y_component = y_component - arg.y_component;
    return temp;
};
vec vec::operator / (double arg){
    vec v_arg;
    v_arg.x_component = x_component/arg;
    v_arg.y_component = y_component/arg;
    return v_arg;
};
vec vec::operator * (double arg){
    vec v_arg;
    v_arg.x_component = x_component*arg;
    v_arg.y_component = y_component*arg;
    return v_arg;
};


double vec::inner_with(vec other){
    return (x_component*other.x_component)+(y_component*other.y_component);
};

double vec::norm(){
    return sqrt(x_component*x_component + y_component*y_component);
};


double vec::distance_wrt_point(vec other){
    return (this->get_X() - other).norm();
};

vec vec::footpoint_wrt(Line arg){
    // consider lines not horizontal or vertical
    double k1,b1,k2,b2;
    vec foot;
    k1 = (arg.get_y2()-arg.get_y1())/(arg.get_x2()-arg.get_x1());
    b1 = arg.get_y1()-arg.get_x1()*k1;
    k2 = -1.0/k1;
    b2 = y_component-x_component*k2;
    foot.set_X((b2-b1)/(k1-k2), k1*(b2-b1)/(k1-k2)+b1); // the intersection of two lines
    return foot;
};

double vec::distance_wrt_line(Line arg){
    if(arg.get_x1() == arg.get_x2()){  // vertical lines
        if((arg.get_y1()-y_component)*(y_component-arg.get_y2()) < 0)  // foot point not in the wall
            return DBL_MAX;
        else
            return abs(x_component-arg.get_x1());
    }
    else if(arg.get_y1() == arg.get_y2()){  // horizontal walls
        if((arg.get_x1()-x_component)*(x_component-arg.get_x2()) < 0)  // foot point not in the wall
            return DBL_MAX;
        else
            return abs(y_component-arg.get_y1());
    }
    else{ // neither vertical nor horizontal walls
        vec foot;
        foot = this->footpoint_wrt(arg);
        if((arg.get_x1()-foot.x_component)*(foot.x_component-arg.get_x2()) < 0) // foot point not in the wall
            return DBL_MAX;
        else
            return this->distance_wrt_point(foot);
    }
};

double vec::touch_point(vec other){
    double distance;
    distance = this->distance_wrt_point(other);
    if(distance > 2.0*_r)
        return 0.0;
    else
        return 2.0*_r - distance;
};

double vec::touch_line(Line arg){
    double distance;
    distance = this->distance_wrt_line(arg);
    if(distance > _r)
        return 0.0;
    else
        return _r - distance;
};

vec vec::direction_from_point(vec other){
    return (this->get_X() - other)/this->distance_wrt_point(other);
};

vec vec::direction_from_line(Line arg){
    vec direction, temp;
    if(arg.get_x1() == arg.get_x2()){  // vertical walls
        if(x_component < arg.get_x1())
            direction.set_X(-1.0, 0.0);
        else
            direction.set_X(1.0, 0.0);
    }
    else if(arg.get_y1() == arg.get_y2()){  // horizontal walls
        if(y_component < arg.get_y1())
            direction.set_X(0.0, -1.0);
        else
            direction.set_X(0.0, 1.0);
    }
    else{
        vec foot;
        foot = this->footpoint_wrt(arg);
        direction = this->get_X().direction_from_point(foot);
    }
    return direction;
};

vec vec::normalvector(){
    vec temp;
    temp.set_X(-y_component, x_component);
    return temp;
};
