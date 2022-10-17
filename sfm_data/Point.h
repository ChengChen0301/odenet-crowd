#ifndef SIMULATION_POINT_H
#define SIMULATION_POINT_H
#include "vec.h"

using namespace std;

class Point{
private:
    vec m_C; // coordinate of the point
    vec m_F; // force on the point
    vec m_F1;
    vec m_F2;
    vec m_F3;
    vec m_A; // acceleration of the point
    vec m_V; // velocity of the point
    int m_G[2]; // grid information of the point
    int flag;// indicate whether the point is still in the room
    double weight; // the weight of the point
public:

    Point()
    {
        m_C.set_X(0.0, 0.0);
        m_F.set_X(0.0, 0.0);
        m_A.set_X(0.0, 0.0);
        m_V.set_X(0.0, 0.0);
        m_F1.set_X(0.0, 0.0);
        m_F2.set_X(0.0, 0.0);
        m_F3.set_X(0.0, 0.0);
        m_G[0] = -1; m_G[1] = -1;
        flag = 1;
        weight = 80.0;
    };

    ///////////////////////
    ///set member data/////
    ///////////////////////
    void set_C(vec arg_C){
        m_C = arg_C;
    };
    void set_V(vec arg_V){
        m_V = arg_V;
    };
    void set_F(vec arg_F){
        m_F = arg_F;
    };
    void set_F1(vec arg_F){
        m_F1 = arg_F;
    };
    void set_F2(vec arg_F){
        m_F2 = arg_F;
    };
    void set_F3(vec arg_F){
        m_F3 = arg_F;
    };
    void set_A(vec arg_A){
        m_A = arg_A;
    };
    void set_G(int arg0, int arg1){
        m_G[0] = arg0;
        m_G[1] = arg1;
    };
    void set_flag(int arg){
        flag = arg;
    };
    void set_weight(double arg_W){
    	weight = arg_W;
    };

    ///////////////////////
    ///visit member data///
    ///////////////////////
    vec get_C(){
        return m_C;
    };
    vec get_V(){
        return m_V;
    };
    vec get_F(){
        return m_F;
    };
    vec get_F1(){
        return m_F1;
    };
    vec get_F2(){
        return m_F2;
    };
    vec get_F3(){
        return m_F3;
    };
    vec get_A(){
        return m_A;
    };
    int *get_G(){
        return m_G;
    };
    int get_flag(){
        return flag;
    };
    double get_weight(){
    	return weight;
    }
};
#endif //SIMULATION_POINT_H
