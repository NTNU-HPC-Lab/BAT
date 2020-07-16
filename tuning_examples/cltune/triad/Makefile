CC=g++
CLTUNE_PATH?=/usr/local/CLTune
CLTUNE_INCLUDE?=$(CLTUNE_PATH)/include
CLTUNE_LIBRARY?=$(CLTUNE_PATH)/build

default:
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@$(CC) -Wall -Wextra -Werror ../cltune_json_saver.cpp triad_tuner.cpp -o triad -I .. -I $(CLTUNE_INCLUDE) -Wl,-rpath $(CLTUNE_LIBRARY)  -L $(CLTUNE_LIBRARY) -lcltune
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) triad
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
