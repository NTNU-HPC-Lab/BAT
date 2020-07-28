CC=g++
CLTUNE_PATH?=/usr/local/CLTune
CLTUNE_INCLUDE?=$(CLTUNE_PATH)/include
CLTUNE_LIBRARY?=$(CLTUNE_PATH)/build

default:
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@$(CC) ../../../src/programs/common/Graph.cpp ../cltune_json_saver.cpp bfs_tuner.cpp -o bfs -I ../../../src/programs/common/ -I .. -I $(CLTUNE_INCLUDE) -Wl,-rpath $(CLTUNE_LIBRARY)  -L $(CLTUNE_LIBRARY) -lcltune
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) bfs
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
