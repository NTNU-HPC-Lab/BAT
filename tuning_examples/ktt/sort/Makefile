CC=g++
KTT_PATH=/usr/local/KTT/source
KTT_LIBRARY=/usr/local/KTT/build/x86_64_Release/

default:
	@echo "\033[93m[i]\033[0m Compiling... 🔨"
	@$(CC) sort.cpp -o sort -I $(KTT_PATH) -L $(KTT_LIBRARY) -lktt
	@echo "\033[92m[✓]\033[0m Done compiling program! 🎉"

clean:
	@echo "\033[93m[i]\033[0m Cleaning up... 🗑️"
	@$(RM) sort
	@echo "\033[92m[✓]\033[0m Done cleaning up! 🎉"
