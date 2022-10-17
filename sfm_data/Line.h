#ifndef SIMULATION_LINE_H
#define SIMULATION_LINE_H
using namespace std;

// to represent a wall
class Line{
private:
    double x1_component;
    double y1_component;
    double x2_component;
    double y2_component;
public:

    Line()
    {
        x1_component = 0.0;
        y1_component = 0.0;
        x2_component = 0.0;
        y2_component = 0.0;
    };

    void set_X(double arg1, double arg2, double arg3, double arg4){
        x1_component = arg1;
        y1_component = arg2;
        x2_component = arg3;
        y2_component = arg4;
    };

    ///////////////////////
    ///visit member data///
    ///////////////////////
    double get_x1(){
        return x1_component;
    }
    double get_y1(){
        return y1_component;
    }
    double get_x2(){
        return x2_component;
    }
    double get_y2(){
        return y2_component;
    }

};
#endif //SIMULATION_LINE_H
