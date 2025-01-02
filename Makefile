# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -std=c++11 -Wall -Wextra -fopenmp

# Chemins d'inclusion
INC = -I CImg_latest/CImg-3.5.0_pre12042411 -I OpenCL-Headers/CL

# Bibliothèques
LIBS = -lOpenCL -lm -lpthread -lX11 -ljpeg

# Nom de l'exécutable
TARGET = exec

# Fichiers source
SRCS = src/main.cpp src/image_processor.cpp src/gaussian_blur_processor.cpp

OBJS = $(SRCS:.cpp=.o)

# Règle par défaut
all: $(TARGET)

# Compilation du programme
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INC) $(SRCS) -o $(TARGET) $(LIBS)

# Nettoyage
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean
