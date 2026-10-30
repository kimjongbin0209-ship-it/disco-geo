"""One environment per process, so a batch of environments steps in parallel.

Workers use the "spawn" start method: forking after JAX has started threads is
unsafe. A worker owns one real environment and returns each timestep together
with the bookkeeping attributes ``EpisodeRecord`` reads (IDs, split,
goal_segment). Renderer settings such as MUJOCO_GL and EGL vendor files are
inherited from the parent's environment variables. This module imports no JAX.
"""

import multiprocessing
import traceback


ATTRIBUTES = ("episode_id", "instance_id", "split", "goal_segment")


def _worker(connection, factory, args):
    env = None
    try:
        env = factory(*args)
        connection.send(("ready", (env.observation_spec(), env.metadata,
                                   {name: getattr(env, name, None) for name in ATTRIBUTES})))
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            timestep = env.reset() if command == "reset" else env.step(value)
            connection.send(("timestep", (timestep,
                                          {name: getattr(env, name) for name in ATTRIBUTES})))
    except Exception:  # Report to the parent instead of dying silently.
        connection.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        connection.close()


class ProcessEnvironment:
    """Environment proxy with the interface BatchedEnvironment and EpisodeRecord use.

    ``reset_async``/``step_async`` followed by ``wait`` let a batch issue every
    command before collecting any result. ``reset``/``step`` are synchronous.
    """

    def __init__(self, factory, *args):
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, factory, args), daemon=True)
        self._process.start()
        child.close()
        self._pending = False
        self._spec, self._metadata, attributes = self._receive("ready")
        self.__dict__.update(attributes)

    @property
    def metadata(self):
        return dict(self._metadata)

    def observation_spec(self):
        return self._spec

    def _receive(self, expected):
        try:
            kind, payload = self._connection.recv()
        except EOFError as exc:
            raise RuntimeError("Environment worker exited unexpectedly") from exc
        if kind == "error":
            raise RuntimeError(f"Environment worker failed:\n{payload}")
        if kind != expected:
            raise RuntimeError(f"Unexpected worker message {kind!r}")
        return payload

    def _send(self, command, value=None):
        if self._pending:
            raise RuntimeError("Wait for the previous command before sending another")
        self._connection.send((command, value))
        self._pending = True

    def reset_async(self):
        self._send("reset")

    def step_async(self, action):
        self._send("step", int(action))

    def wait(self):
        if not self._pending:
            raise RuntimeError("No command is pending")
        self._pending = False
        timestep, attributes = self._receive("timestep")
        self.__dict__.update(attributes)
        return timestep

    def reset(self):
        self.reset_async()
        return self.wait()

    def step(self, action):
        self.step_async(action)
        return self.wait()

    def close(self):
        if self._process.is_alive():
            try:
                self._connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            self._process.join(timeout=10)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._connection.close()
