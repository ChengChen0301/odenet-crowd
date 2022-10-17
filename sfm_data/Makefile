CXX = g++   
CXXFLAGS= -lboost_program_options 
LIBS= -I/opt/software/boost-1-58-0/include/ -I/usr/include/gsl -L/opt/software/boost-1-58-0/lib/ -L/usr/lib64/ -lm -lgsl -lgslcblas


prom = test
deps = simulation.h vec.h variables.h 
obj = main.o simulation.o vec.o variables.o


$(prom): $(obj)
	$(CXX) -o $(prom) $(obj)

%.o: %.c $(deps)
	$(CXX) -c $< -o $@



