MMF:
	gcc -o MMF MMF.c -lpthread -lm
	./MMF test-noise.png noise-output-test.png 3 4

DDF:	
	gcc -o DDF DDF.c -lpthread -lm
	./DDF test-soft.png soft-output-test.png 10 50.0 4
