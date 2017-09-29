package display;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL11.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.NULL;

import java.nio.IntBuffer;

import org.lwjgl.glfw.GLFWErrorCallback;
import org.lwjgl.glfw.GLFWVidMode;
import org.lwjgl.opengl.GL;
import org.lwjgl.system.MemoryStack;

import map.TileType;

public class Window {

	// The window handle
	private long window;

	private Renderer r;
	private int width, height;
	private boolean vsync = true;
	private boolean render = true;
	private long counter = 10;

	public Window(int width, int height) {
		this.width = width;
		this.height = height;
		init();
	}

	public void init() {
		// Setup an error callback. The default implementation
		// will print the error message in System.err.
		GLFWErrorCallback.createPrint(System.err).set();

		// Initialize GLFW. Most GLFW functions will not work before doing this.
		if (!glfwInit())
			throw new IllegalStateException("Unable to initialize GLFW");

		// Configure GLFW
		glfwDefaultWindowHints(); // optional, the current window hints are
									// already the default
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // the window will stay hidden
													// after creation
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // the window will be
													// resizable

		// Create the window
		window = glfwCreateWindow(width, height, "GA DEMO", NULL, NULL);
		if (window == NULL)
			throw new RuntimeException("Failed to create the GLFW window");

		// Setup a key callback. It will be called every time a key is pressed,
		// repeated
		// or released.
		glfwSetKeyCallback(window, (window, key, scancode, action, mods) -> {
			if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE)
				glfwSetWindowShouldClose(window, true); // We will detect this
														// in the rendering loop
		});

		// Get the thread stack and push a new frame
		try (MemoryStack stack = stackPush()) {
			IntBuffer pWidth = stack.mallocInt(1); // int*
			IntBuffer pHeight = stack.mallocInt(1); // int*

			// Get the window size passed to glfwCreateWindow
			glfwGetWindowSize(window, pWidth, pHeight);

			// Get the resolution of the primary monitor
			GLFWVidMode vidmode = glfwGetVideoMode(glfwGetPrimaryMonitor());

			// Center the window
			glfwSetWindowPos(window, (vidmode.width() - pWidth.get(0)) / 2, (vidmode.height() - pHeight.get(0)) / 2);
		} // the stack frame is popped automatically

		// Make the OpenGL context current
		glfwMakeContextCurrent(window);
		// Enable v-sync
		glfwSwapInterval(1);

		// Make the window visible
		glfwShowWindow(window);

		// This line is critical for OpenGL's inter-operation with GLFW's
		// OpenGL context, or any context that is managed externally.
		// LWJGL detects the context that is current in the current thread,
		// creates the GLCapabilities instance and makes the OpenGL
		// bindings available for use.
		GL.createCapabilities();

		// Set the clear color
		glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
		r = new Renderer(width, height);

		// Setup resize callback
		glfwSetFramebufferSizeCallback(window, (window, width, height) -> {
			this.width = width;
			this.height = height;
			r.resize(width, height);
			glViewport(0, 0, width, height);
		});
	}

	public void destroy() {
		r.cleanup();
		// Free the window callbacks and destroy the window
		glfwFreeCallbacks(window);
		glfwDestroyWindow(window);

		// Terminate GLFW and free the error callback
		glfwTerminate();
		glfwSetErrorCallback(null).free();
	}

	public long getHandle() {
		return window;
	}

	public void refresh() {
		if(glfwGetKey(window, GLFW_KEY_V) == 1 && counter <= System.currentTimeMillis()) {
			vsync = !vsync;
			glfwSwapInterval(vsync?1:0);
			counter = System.currentTimeMillis()+200;
		}
		glfwSwapBuffers(window);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the
															// framebuffer
		// Poll for window events. The key callback above will only be
		// invoked during this call.
		glfwPollEvents();
	}

	public boolean shouldClose() {
		return glfwWindowShouldClose(window);
	}

	public void renderTile(float x, float y, float size, TileType tile) {
		if(glfwGetKey(window, GLFW_KEY_R) == 1 && counter <= System.currentTimeMillis()) {
			render = !render;
			counter = System.currentTimeMillis()+200;
		}
		if(render){
		switch (tile.name()) {
		case "FOOD":
			r.render((float) (x + 0.25) * width / size, (float) (y + 0.25) * height / size, width / (size * 2),
					height / (size * 2), tile.getColor());
			break;
		case "ENTITY":
		case "WALL":
			r.render(x * width / size, y * height / size, width / size, height / size, tile.getColor());
			break;
		case "EMPTY":
		default:
		}
		}
	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

}
