/*
 * Circle.cpp
 * RVO2 Library
 *
 * Copyright 2008 University of North Carolina at Chapel Hill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please send all bug reports to <geom@cs.unc.edu>.
 *
 * The authors may be contacted via:
 *
 * Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha
 * Dept. of Computer Science
 * 201 S. Columbia St.
 * Frederick P. Brooks, Jr. Computer Science Bldg.
 * Chapel Hill, N.C. 27599-3175
 * United States of America
 *
 * <http://gamma.cs.unc.edu/RVO2/>
 */

/*
 * Example file showing a demo with 250 agents initially positioned evenly
 * distributed on a circle attempting to move to the antipodal position on the
 * circle.
 */

#ifndef RVO_OUTPUT_TIME_AND_POSITIONS
#define RVO_OUTPUT_TIME_AND_POSITIONS 1
#endif

#include <cmath>
#include <cstddef>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <cstring>
#include <fstream>
#include "sys/timeb.h"

#if RVO_OUTPUT_TIME_AND_POSITIONS
#include <iostream>
#endif

#if _OPENMP
#include <omp.h>
#endif

#include <RVO.h>

#ifndef M_PI
const float M_PI = 3.14159265358979323846f;
#endif


#define _right (5.0)
#define _left (-5.0)
#define _up (5.0)
#define _down (-5.0)
#define _r (0.32)

#define _right_ (0.0)
#define _left_ (-5.0)
#define _up_ (5.0)
#define _down_ (0.0)
#define _width (1.0)

#define width_ (100.0)
#define height_ (100.0)
#define Radius (0.32)
#define Max_speed (1.0)

using namespace std;

/* Store the goals of the agents. */
std::vector<RVO::Vector2> goals;
std::vector<RVO::Vector2> temps;

void setupScenario(RVO::RVOSimulator *sim)
{
	/* Specify the global time step of the simulation. */
	sim->setTimeStep(0.01);
	double inner_width = width_*0.8;
	double inner_height = height_*0.8;
	double wall_width = Radius*1.5;
	double half_inner_width = inner_width / 2;
	double half_inner_height = inner_height / 2;
	double half_exit_width = Radius*2.0;

	/* Specify the default parameters for agents that are subsequently added. */
	// neighborDist, maxNeighbors, timeHorizon, timeHorizonObst, radius, maxSpeed, const Vector2 &velocity
	sim->setAgentDefaults(2.5, 10, 0.05, 0.05, Radius, Max_speed, RVO::Vector2(0.0, 0.0));

	/*
	 * Add agents, specifying their start position, and store their goals on the
	 * opposite side of the environment.
	 */
	double x = (double)rand() / RAND_MAX *(_right_ - _left_ - 2.0*_r) + _left_ + _r; // the first particle
    double y = (double)rand() / RAND_MAX *(_up_ - _down_ - 2.0*_r) + _down_ + _r;
    sim->addAgent(RVO::Vector2(x, y));

	for (int i = 1; i < 20; i++) {
		int flag = 1;
		while(flag == 1){
            x = (double)rand() / RAND_MAX *(_right_ - _left_ - 2.0*_r) + _left_ + _r;
            y = (double)rand() / RAND_MAX *(_up_ - _down_ - 2.0*_r) + _down_ + _r;
            temps.push_back(RVO::Vector2(x, y));
            int index = 0;
            for (int j=0; j<i; j++){
            	if(RVO::absSq(sim->getAgentPosition(j) - temps[0]) <= 4.0*_r*_r){
            		index = j;
            		break;
            	}
            	index = j + 1;
            }
            if(index >= i && RVO::absSq(sim->getAgentPosition(i-1) - temps[0]) > 4.0*_r*_r){
            	flag = 0;
            	sim->addAgent(RVO::Vector2(x, y));
            }
            temps.pop_back();
        }
	}
	// cout << "here\n";
	// goals.push_back(RVO::Vector2(half_inner_width + wall_width + 5.0, 0.0));
	goals.push_back(RVO::Vector2(_right+2*Radius, 0.0));
	// goals.push_back(RVO::Vector2(100.0f, 0.0f));

	std::vector<RVO::Vector2> obstacle1, obstacle2, obstacle3, obstacle4, obstacle5, obstacle6, obstacle7;
	obstacle1.push_back(RVO::Vector2(_left, _up));
	obstacle1.push_back(RVO::Vector2(_left, _down));

	obstacle2.push_back(RVO::Vector2(_left, _up));
	obstacle2.push_back(RVO::Vector2(_right, _up));

	obstacle3.push_back(RVO::Vector2(_right, _up));
	obstacle3.push_back(RVO::Vector2(_right, _width));
	// obstacle3.push_back(RVO::Vector2(_right + 1.0, _width));
	// obstacle3.push_back(RVO::Vector2(_right + 1.0, _up));

	obstacle4.push_back(RVO::Vector2(_right, -_width));
	obstacle4.push_back(RVO::Vector2(_right, _down));
	// obstacle4.push_back(RVO::Vector2(_right + 1.0, _down));
	// obstacle4.push_back(RVO::Vector2(_right + 1.0, -_width));


	obstacle5.push_back(RVO::Vector2(_left, _down));
	obstacle5.push_back(RVO::Vector2(_right, _down));

	sim->addObstacle(obstacle1);
	sim->addObstacle(obstacle2);
	sim->addObstacle(obstacle3);
	sim->addObstacle(obstacle4);
	sim->addObstacle(obstacle5);

	sim->processObstacles();
}

#if RVO_OUTPUT_TIME_AND_POSITIONS
void updateVisualization(RVO::RVOSimulator *sim, string folder, int t)
{
	/* Output the current global time. */
	// std::cout << sim->getGlobalTime();
	std::string filename1 = folder + "/coord" + to_string(t) + ".txt";
	std::string filename2 = folder + "/velocity" + to_string(t) + ".txt";
	ofstream fout;
	fout.open(filename1.c_str());
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		fout << sim->getAgentPosition(i).x() << " " << sim->getAgentPosition(i).y() << "\n";
	}
	fout<<flush;fout.close();
	fout.open(filename2.c_str());
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		fout << sim->getAgentVelocity(i).x() << " " << sim->getAgentVelocity(i).y() << "\n";
	}
	fout<<flush;fout.close();
	/* Output the current position of all the agents. */
}
#endif

void setPreferredVelocities(RVO::RVOSimulator *sim)
{
	double wall_width = Radius*1.5;
	double inner_width = width_*0.8;
	double half_inner_width = inner_width / 2;
	/*
	 * Set the preferred velocity to be a vector of unit magnitude (speed) in the
	 * direction of the goal.
	 */
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < static_cast<int>(sim->getNumAgents()); ++i) {

		RVO::Vector2 goalVector = goals[0] - sim->getAgentPosition(i);
		if(sim->getAgentPosition(i).x() > _right)
			goalVector = RVO::Vector2(1.0, 0.0);

		// if(RVO::absSq(goalVector) < 0.3)
			// goalVector = goals[1] - sim->getAgentPosition(i);

		goalVector = RVO::normalize(goalVector);
		sim->setAgentPrefVelocity(i, goalVector*Max_speed);
	}
}

bool reachedGoal(RVO::RVOSimulator *sim)
{
	double inner_width = width_*0.8;
	double half_inner_width = inner_width / 2;
	/* Check if all agents have reached their goals. */
	for (size_t i = 0; i < sim->getNumAgents(); ++i) {
		if (sim->getAgentPosition(i).x() < _right) {
			return false;
		}
	}

	return true;
}

int main(int argc, char *argv[])
{
	/* Create a new simulator instance. */
	RVO::RVOSimulator *sim = new RVO::RVOSimulator();
	string folder = "near_20_2";
	mkdir(folder.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
	char *k = argv[1];
	folder += "/round";
	folder += k;
	mkdir(folder.c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRWXG | S_IRWXO);
	/* Set up the scenario. */
	srand(time(NULL));
	struct timeb timeSeed;
    ftime(&timeSeed);
    srand(timeSeed.time * 1000 + timeSeed.millitm); 
	setupScenario(sim);
	int t = 0;
	updateVisualization(sim, folder, t);


// 	/* Perform (and manipulate) the simulation. */
	do {
#if RVO_OUTPUT_TIME_AND_POSITIONS
#endif
		setPreferredVelocities(sim);
		sim->doStep();
		t = t + 1;
		updateVisualization(sim, folder, t);
		// cout << t << "\n";
	}
	while (!reachedGoal(sim));
	// while (t < 1000);

	delete sim;
	cout << k << "\n";
	return 0;
}
