#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "sys/timeb.h"

#include "vec.h"
#include "Point.h"
#include "variables.h"
#include "Line.h"
#include "simulation.h"

using namespace std;

int main(int argc, char *argv[]){

    char *k = argv[1];
    string dir = "samples80";
    mkdir(dir.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
    c = 0.0;
    srand(time(NULL));
    struct timeb timeSeed;
    ftime(&timeSeed);
    srand(timeSeed.time * 1000 + timeSeed.millitm); 

    //-----------------------create the folders------------------------
    string folder;
    folder += dir;
    folder += "/round";
    folder += k;
    mkdir(folder.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);

    string PosName = folder + "/coord.txt";
    string VelName = folder + "/velocity.txt";
    string ForceName = folder + "/force.txt";
    string MassName = folder + "/mass.txt";

    string ETName = folder + "/evacuation_time.txt";
    string OPName = folder + "/outside_point.txt";
    //-----------------------create the folders------------------------
    initialize();
    
    computeForce();
    int t=0;
    savedata(t);

    while(number != 0){
        t++;
        update(t);
        computeForce();
        if(t%10 == 0){ // record every 0.1 seconds
            savedata(t/10);
        }
    }

    // ofstream fout;
    ofstream write;
    ifstream read;
    // fout.open(ETName.c_str());
    // for(int i=0; i<Np; i++)
    //     fout<<left<<setw(7)<<ET[i]<<"\n";
    // fout<<left<<setw(7)<<t<<"\n"; // total evacuation time
    // fout<<flush;fout.close();

   //-----------------------to save the positions every other 0.1s in a .txt file------------------------
    // for(int j=0; j<=t/100; j++) {
    //     string index = to_string(j*100);
    //     string PosName = folder + "/output" + index + ".txt";
    //     fout.open(PosName.c_str());
    //     for (int i = 0; i < Np; i++) {
    //         fout << left << setw(13) << save_position[i][0][j] << '\t' << left << setw(13) << save_position[i][1][j] <<"\n";
    //              // << '\t' << left << setw(3) << save_flag[j][i] << "\n";
    //     }
    //     fout<<flush;fout.close();
    // }

    //-----------------------to save all the positions in a .txt file------------------------
    int nFrame = t/10;

    write.open(PosName.c_str(), ios::app);
    for(int j=0; j<nFrame; j++){
        for(int i=0; i<Np; i++)
            // fout<<left<<setw(13)<<save_position[i][0][j]<<'\t'<<left<<setw(13)<<save_position[i][1][j]<<'\t'<<left<<setw(3)<<_up<<'\t';
        // fout<<left<<setw(13)<<save_position[Np-1][0][j]<<'\t'<<left<<setw(13)<<save_position[Np-1][1][j]<<'\t'<<left<<setw(3)<<_up<<"\n";
            write<<left<<setw(13)<<save_position[i][0][j]<<'\t'<<left<<setw(13)<<save_position[i][1][j]<<'\n';
        // write<<left<<setw(13)<<save_position[Np-1][0][j]<<'\t'<<left<<setw(13)<<save_position[Np-1][1][j]<<"\n";
    }
    // fout<<flush;
    write.close();

    write.open(VelName.c_str(), ios::app);
    for(int j=0; j<nFrame; j++){
        for(int i=0; i<Np; i++)
            // fout<<left<<setw(13)<<save_force[i][0][j]<<'\t'<<left<<setw(13)<<save_force[i][1][j]<<'\t'<<left<<setw(3)<<0<<'\t';
        // fout<<left<<setw(13)<<save_force[Np-1][0][j]<<'\t'<<left<<setw(13)<<save_force[Np-1][1][j]<<'\t'<<left<<setw(3)<<0<<"\n";
            write<<left<<setw(13)<<save_velocity[i][0][j]<<'\t'<<left<<setw(13)<<save_velocity[i][1][j]<<'\n';
    }
    // fout<<flush;
    write.close();

    write.open(MassName.c_str(), ios::app);
    for(int i=0; i<Np; i++){
        write<<left<<setw(13)<<pt[i].get_weight()<<'\n';
    }
    // fout<<flush;
    write.close();
    
    // write.open(ForceName.c_str(), ios::app);
    // for(int j=0; j<nFrame; j++){
    //     for(int i=0; i<Np; i++)
    //         write<<left<<setw(13)<<save_force[i][0][j]<<'\t'<<left<<setw(13)<<save_force[i][1][j]<<'\t'<<left<<setw(13)<<save_force[i][2][j]
    //     <<'\t'<<left<<setw(13)<<save_force[i][3][j]<<'\t'<<left<<setw(13)<<save_force[i][4][j]<<'\t'<<left<<setw(13)<<save_force[i][5][j]
    //     <<'\t'<<left<<setw(13)<<save_force[i][6][j]<<'\t'<<left<<setw(13)<<save_force[i][7][j]<<'\n';
    // }
    // write.close();
    read.close();

    // fout.open(OPName.c_str());
    // for(int j=0; j<=t/100; j++)
    //     fout<<left<<setw(7)<<j*100<<'\t'<<left<<setw(7)<<Out_number[j]<<"\n";
    // fout<<flush;fout.close();
    cout<<"The "<<k<<"th running ends"<<"\n";
    return 0;
}

