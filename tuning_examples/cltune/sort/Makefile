CC=g++
CLTUNE_PATH=/usr/local/CLTune/include
CLTUNE_LIBRARY=/usr/local/CLTune/build

default:
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@$(CC) -Wall -Wextra -Werror sort.cpp -o sort -I $(CLTUNE_PATH) -L $(CLTUNE_LIBRARY) -lcltune
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) sort
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
