package map;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import org.joml.Vector2i;

import display.Window;
import entities.ReinforcementEntity;
import logging.Logger;
import logging.Logger.Category;

public class TileMap {

	private Set<ReinforcementEntity> livingEntities;
	private TileType[][] tiles;
	private int size;

	public TileMap(int size) {
		this.size = size;
		livingEntities = new HashSet<>();
		tiles = new TileType[size][size];
	}

	public void reset() {
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) {
				if ((x == 0 || x == size - 1) || (y == 0 || y == size - 1)) {
					tiles[x][y] = TileType.WALL;
				} else {
					tiles[x][y] = TileType.EMPTY;
				}
			}
		}
	}

	public void spawnPopulation(Population population) {
		for (ReinforcementEntity entity : population.getEntities()) {
			Vector2i emptyTileLoc = getEmptyTile();
			entity.setPosition(emptyTileLoc);
			setTile(emptyTileLoc, TileType.ENTITY);
			livingEntities.add(entity);
			entity.reset();
		}
	}

	public Vector2i getEmptyTile() {
		Random rand = new Random();
		int x;
		int y;

		do {
			x = rand.nextInt(size);
			y = rand.nextInt(size);
		} while (!getTile(x, y).equals(TileType.EMPTY));
		return new Vector2i(x, y);
	}

	public void update() {
		for (Iterator<ReinforcementEntity> iter = livingEntities.iterator(); iter.hasNext();) {
			ReinforcementEntity entity = iter.next();
			if (!entity.isDead()) {
				entity.update(this);
			}
			if (entity.isDead()) {
				setTile(entity.getPosition(), TileType.EMPTY);
				iter.remove();
			}
		}
	}

	public void killEntity(Vector2i entityPosition) {
		for (Iterator<ReinforcementEntity> iter = livingEntities.iterator(); iter.hasNext();) {
			ReinforcementEntity entity = iter.next();
			if (entityPosition.equals(entity.getPosition())) {
				entity.kill();
			}
		}
		Logger.debug("Entity Killed", Category.ENTITIES);
	}

	public void setEmptyTile(TileType type) {
		Vector2i emptyTile = getEmptyTile();
		tiles[emptyTile.x()][emptyTile.y()] = type;
	}

	public TileType getTile(int x, int y) {
		return tiles[x][y];
	}

	public void setTile(int x, int y, TileType type) {
		tiles[x][y] = type;
	}

	public void setTile(Vector2i tile, TileType type) {
		setTile(tile.x(), tile.y(), type);
	}

	public TileType[][] getTiles() {
		return tiles;
	}

	public Set<ReinforcementEntity> getEntities() {
		return livingEntities;
	}

	public void render(Window w) {
		for (int y = 0; y < size; y++) {
			for (int x = 0; x < size; x++) {
				w.renderTile(x, y, size, tiles[x][y]);
			}
		}
	}

	public boolean outOfRange(int x, int y) {
		if ((x < 0 || x >= size) || (y < 0 || y >= size)) {
			return true;
		}
		return false;
	}

	public boolean outOfRange(Vector2i point) {
		return outOfRange(point.x(), point.y());
	}
}
