package com.masterapp;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

public class EditorStateManager {
    private static final String STATE_FILE = "editor_state.ser";
    private Map<String, Object> state;

    public EditorStateManager() {
        state = new HashMap<>();
    }

    public void saveState() {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(STATE_FILE))) {
            oos.writeObject(state);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void restoreState() {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(STATE_FILE))) {
            state = (Map<String, Object>) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void setState(String key, Object value) {
        state.put(key, value);
    }

    public Object getState(String key) {
        return state.get(key);
    }

    public void clearState() {
        state.clear();
    }
}