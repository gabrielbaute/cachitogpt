from pyfiglet import Figlet
from colorama import Fore, Style, init

init(autoreset=True)  # Inicializa colorama para soporte de colores en Windows

class Banner:
    """Banner de presentacion de la app"""
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.description = f"Sistema Generativo Local {name}"
        self.banner = self.get_banner()
        
    def _build_title(self, name: str) -> str:
        """Genera el título en arte ASCII usando pyfiglet.
        
        Args:
            name (str): Nombre de la aplicación.
        
        Returns:
            str: Título en arte ASCII.
        """
        figlet = Figlet(font="ansi_shadow", width=120, justify="center")
        ascii_art = figlet.renderText(name)
        return ascii_art

    def get_banner(self) -> str:
        """Devuelve el banner como una cadena de texto.
        
        Returns:
            str: Banner en arte ASCII.
        """
        title = self._build_title(self.name)
        banner = f'''{Fore.CYAN}{Style.BRIGHT}\n{title}\n{self.description} [v{self.version}]'''
        return banner
    
    def print_banner(self) -> None:
        """Imprime el banner en la consola.
        
        Returns:
            None
        """
        banner = self.banner
        print(banner)
        