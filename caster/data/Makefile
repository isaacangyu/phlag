dstar: dir
	g++ -std=gnu++11 -march=native -Ofast dstar.cpp -o bin/dstar || g++ -std=gnu++17 -O2 dstar.cpp -o bin/dstar

dir:
	g++ -v 2>&1 | tail -n 1
	echo 'If installation failed, please ensure g++ version >= 7.5.0'
	mkdir -p bin
	
clean:
	rm bin/*