CC=g++
CLTUNE_PATH=/usr/local/CLTune/include
CLTUNE_LIBRARY=/usr/local/CLTune/build

default:
	@$(CC) -Wall -Wextra -Werror sort.cpp -o sort -I $(CLTUNE_PATH) -L $(CLTUNE_LIBRARY) -lcltune

clean:
	@$(RM) sort
