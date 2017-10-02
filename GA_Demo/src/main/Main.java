package main;

import display.Window;
import logging.Logger;
import map.Village;

public class Main {

	private static final int POPULATION_SIZE = 1;
	private static final int MAP_SIZE = 50;
	private static final int NUM_FOOD = 80;

	private Village village;

	private static final int SCREEN_WIDTH = 600;
	private static final int SCREEN_HEIGHT = 600;

	Window w;

	public void run() {
		setup();
		while (!w.shouldClose()) {
			update();
			render();
		}
	}

	public void setup() {
		Logger.info("Setup Start");

		w = new Window(SCREEN_WIDTH, SCREEN_HEIGHT);

		village = new Village(POPULATION_SIZE, MAP_SIZE, NUM_FOOD);
		village.setUp();

		Logger.info("Setup End");
	}

	public void update() {
		village.update();
	}

	public void render() {
		w.refresh();
		village.render(w);
	}

	public static void main(String[] args) {
		Logger.info("Start");
		Main main = new Main();
		main.run();
	}

}
