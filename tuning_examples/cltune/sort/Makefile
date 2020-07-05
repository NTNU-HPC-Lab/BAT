CC=g++
CLTUNE_PATH=~/Desktop/auto-tuners/CLTune/include
CLTUNE_LIBRARY=~/Desktop/auto-tuners/CLTune/build

default:
	@$(CC) -Wall -Wextra -Werror sort.cpp -o sort -I $(CLTUNE_PATH) -L $(CLTUNE_LIBRARY) -lcltune

clean:
	@$(RM) sort
