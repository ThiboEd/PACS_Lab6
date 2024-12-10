# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -std=c++11 -Wall -Wextra

# Chemins d'inclusion
INC = -I CImg_latest/CImg-3.5.0_pre12042411 -I OpenCL-Headers/CL

# Bibliothèques
LIBS = -lOpenCL -lm -lpthread -lX11 -ljpeg

# Nom de l'exécutable
TARGET = gaussianK

# Fichiers source
SRCS = main.cpp

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
