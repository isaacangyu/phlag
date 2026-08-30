dstar: dir
	g++ -std=gnu++11 -march=native -Ofast dstar.cpp -o bin/dstar || g++ -std=gnu++17 -O2 dstar.cpp -o bin/dstar

caster-pair: dir
	g++ -std=gnu++11 -march=native -Ofast -Iincludes caster-pair.cpp -o bin/caster-pair || g++ -std=gnu++17 -O2 -Iincludes caster-pair.cpp -o bin/caster-pair

caster-site: dir
	g++ -std=gnu++11 -march=native -Ofast -Iincludes caster-site.cpp -o bin/caster-site || g++ -std=gnu++17 -O2 -Iincludes caster-site.cpp -o bin/caster-site

dir:
	g++ -v 2>&1 | tail -n 1
	echo 'If installation failed, please ensure g++ version >= 7.5.0'
	mkdir -p bin
	
clean:
	rm bin/*