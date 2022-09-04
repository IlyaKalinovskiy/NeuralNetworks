import abc


class BaseAbstractWorker:
    """Basic interface for a worker class."""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def deactivate(self):
        """Sets execution flag to false.

        join() should be called separately.

        """

    @abc.abstractmethod
    def is_active(self):
        """Return the execution flag value.

        :return: the execution flag value.

        """
        pass

    @abc.abstractmethod
    def do_work_once(self):
        """Implement a single iteration of the main loop here."""
        pass

    def on_start(self):
        """Implement in a derivative class, if you wish to do something before the main
        loop."""
        pass

    def on_crash(self):
        """Implement in a derivative class, if you wish to perform some operations in the
        case of an unhandled exception.

        You may use sys.exc_info() to get stacktrace.

        """
        pass

    def on_finish(self):
        """Implement in a derivative class, if you wish to do something."""
        pass

    def _run(self):
        """This method should be run in the new thread/process.

        In case of standard threads/processes, it should be called from run()
        function.

        """
        try:
            self.on_start()
            while self.is_active():
                self.do_work_once()
        except:
            self.on_crash()
        finally:
            self.on_finish()
