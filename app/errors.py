class ModelUnavailableError(RuntimeError):
    """Модель недоступна или вернула некорректный результат."""

class WardrobeUnavailableError(RuntimeError):
    """Не удалось прочитать гардероб."""
