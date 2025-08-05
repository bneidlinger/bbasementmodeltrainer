"""
Retro industrial theme for B's BasementBrewAI
Industrial orange meets IBM green - terminal style!
"""

import dearpygui.dearpygui as dpg

# Color palette
COLORS = {
    # Industrial Orange (main accent)
    'orange_bright': (255, 140, 0),      # #FF8C00 - Bright industrial orange
    'orange_normal': (255, 165, 0),      # #FFA500 - Standard orange
    'orange_dim': (205, 133, 63),        # #CD853F - Dimmed orange
    
    # IBM Green (secondary accent)
    'green_bright': (0, 255, 127),       # #00FF7F - Bright terminal green
    'green_normal': (50, 205, 50),       # #32CD32 - IBM green
    'green_dim': (34, 139, 34),          # #228B22 - Forest green
    
    # Background colors
    'bg_dark': (15, 15, 15),             # Almost black
    'bg_medium': (25, 25, 25),           # Dark gray
    'bg_light': (40, 40, 40),            # Medium gray
    
    # Text colors
    'text_primary': (255, 140, 0),       # Orange for headers
    'text_secondary': (50, 205, 50),     # Green for data
    'text_normal': (200, 200, 200),      # Light gray for regular text
    'text_dim': (128, 128, 128),         # Dimmed text
    
    # Status colors
    'error': (255, 69, 0),               # Red-orange
    'success': (0, 255, 127),            # Bright green
    'warning': (255, 215, 0),            # Gold
}

ASCII_TITLE = """
╔══════════════════════════════════════════════════════════════╗
║  ____  _     ____                                 _          ║
║ | __ )( )___|  _ \\ __ _ ___  ___ _ __ ___   ___ | |_ ___    ║
║ |  _ \\|// __| |_) / _` / __|/ _ \\ '_ ` _ \\ / _ \\| __/ __|   ║
║ | |_) | \\__ \\  _ < (_| \\__ \\  __/ | | | | |  __/| |_\\__ \\   ║
║ |____/  |___/_| \\_\\__,_|___/\\___|_| |_| |_|\\___| \\__|___/   ║
║                                                              ║
║  ____                        _   ____                   _    ║
║ | __ ) _ __ _____      __   / \\ |_ _|                  | |   ║
║ |  _ \\| '__/ _ \\ \\ /\\ / /  / _ \\ | |                   | |   ║
║ | |_) | | |  __/\\ V  V /  / ___ \\| |                   |_|   ║
║ |____/|_|  \\___| \\_/\\_/  /_/   \\_\\_|                   (_)   ║
║                                                              ║
║         [ Industrial ML Training Terminal v1.0 ]             ║
╚══════════════════════════════════════════════════════════════╝
"""

ASCII_TITLE_COMPACT = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║ ██████╗  ██╗███████╗    ██████╗  █████╗ ███████╗███████╗███╗   ███╗███████╗  ║
║ ██╔══██╗ ╚═╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗ ████║██╔════╝  ║
║ ██████╔╝     ███████╗    ██████╔╝███████║███████╗█████╗  ██╔████╔██║█████╗    ║
║ ██╔══██╗     ╚════██║    ██╔══██╗██╔══██║╚════██║██╔══╝  ██║╚██╔╝██║██╔══╝    ║
║ ██████╔╝     ███████║    ██████╔╝██║  ██║███████║███████╗██║ ╚═╝ ██║███████╗  ║
║ ╚═════╝      ╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝  ║
║                          ███╗   ██╗████████╗                                  ║
║                          ████╗  ██║╚══██╔══╝                                  ║
║                          ██╔██╗ ██║   ██║                                     ║
║                          ██║╚██╗██║   ██║                                     ║
║                          ██║ ╚████║   ██║                                     ║
║                          ╚═╝  ╚═══╝   ╚═╝                                     ║
║ ██████╗ ██████╗ ███████╗██╗    ██╗    █████╗ ██╗                             ║
║ ██╔══██╗██╔══██╗██╔════╝██║    ██║   ██╔══██╗██║                             ║
║ ██████╔╝██████╔╝█████╗  ██║ █╗ ██║   ███████║██║                             ║
║ ██╔══██╗██╔══██╗██╔══╝  ██║███╗██║   ██╔══██║██║                             ║
║ ██████╔╝██║  ██║███████╗╚███╔███╔╝   ██║  ██║██║                             ║
║ ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝    ╚═╝  ╚═╝╚═╝                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║             [ INDUSTRIAL ML TRAINING TERMINAL v1.0 ]                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

ASCII_TITLE_80S = """
 ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
 █                                                                           █
 █  ██████╗ ██╗███████╗    ██████╗  █████╗ ███████╗███████╗███╗   ███╗███████╗███╗   ██╗████████╗
 █  ██╔══██╗╚═╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝
 █  ██████╔╝   ███████╗    ██████╔╝███████║███████╗█████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   
 █  ██╔══██╗   ╚════██║    ██╔══██╗██╔══██║╚════██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   
 █  ██████╔╝   ███████║    ██████╔╝██║  ██║███████║███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   
 █  ╚═════╝    ╚══════╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   
 █                                                                           █
 █        ██████╗ ██████╗ ███████╗██╗    ██╗     █████╗ ██╗                █
 █        ██╔══██╗██╔══██╗██╔════╝██║    ██║    ██╔══██╗██║                █
 █        ██████╔╝██████╔╝█████╗  ██║ █╗ ██║    ███████║██║                █
 █        ██╔══██╗██╔══██╗██╔══╝  ██║███╗██║    ██╔══██║██║                █
 █        ██████╔╝██║  ██║███████╗╚███╔███╔╝    ██║  ██║██║                █
 █        ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚══╝╚══╝     ╚═╝  ╚═╝╚═╝                █
 █                                                                           █
 █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀[ Industrial ML Training Terminal ]▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
 ▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀
"""

def create_retro_theme():
    """Create and return the retro industrial theme."""
    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            # Window styling
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)  # Sharp corners
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 1)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 1)
            
            # Main colors
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, COLORS['bg_dark'])
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, COLORS['bg_medium'])
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, COLORS['bg_medium'])
            dpg.add_theme_color(dpg.mvThemeCol_Border, COLORS['green_dim'])
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow, (0, 0, 0, 0))
            
            # Frame colors
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, COLORS['bg_light'])
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (*COLORS['orange_dim'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (*COLORS['orange_normal'], 100))
            
            # Title bar
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg, COLORS['bg_medium'])
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (*COLORS['green_dim'], 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, COLORS['bg_dark'])
            
            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, (*COLORS['green_dim'], 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS['green_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLORS['green_bright'])
            
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS['text_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, COLORS['text_dim'])
            
            # Headers
            dpg.add_theme_color(dpg.mvThemeCol_Header, (*COLORS['orange_dim'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (*COLORS['orange_normal'], 150))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, COLORS['orange_bright'])
            
            # Checkbox/Radio
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark, COLORS['green_bright'])
            
            # Slider/Drag
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, COLORS['orange_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, COLORS['orange_bright'])
            
            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, COLORS['bg_dark'])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, COLORS['green_dim'])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, COLORS['green_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, COLORS['green_bright'])
            
            # Separator
            dpg.add_theme_color(dpg.mvThemeCol_Separator, COLORS['green_dim'])
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered, COLORS['orange_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive, COLORS['orange_bright'])
            
            # Tab
            dpg.add_theme_color(dpg.mvThemeCol_Tab, (*COLORS['green_dim'], 150))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, COLORS['green_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, COLORS['orange_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused, (*COLORS['green_dim'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive, (*COLORS['orange_dim'], 150))
            
            # Plot colors (commented out - not available in current DPG version)
            # dpg.add_theme_color(dpg.mvThemeCol_PlotBg, COLORS['bg_dark'])
            # dpg.add_theme_color(dpg.mvThemeCol_PlotBorder, COLORS['green_dim'])
            # dpg.add_theme_color(dpg.mvThemeCol_PlotLegendBg, (*COLORS['bg_medium'], 200))
            # dpg.add_theme_color(dpg.mvThemeCol_PlotLegendBorder, COLORS['green_dim'])
            # dpg.add_theme_color(dpg.mvThemeCol_PlotLines, COLORS['orange_bright'])
            # dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered, COLORS['orange_normal'])
            
            # Table colors
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg, (*COLORS['green_dim'], 150))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong, COLORS['green_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight, COLORS['green_dim'])
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt, (*COLORS['bg_light'], 50))
    
    return theme

def create_button_themes():
    """Create special themes for different button types."""
    themes = {}
    
    # Start/Success button
    with dpg.theme() as themes['start']:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (*COLORS['green_dim'], 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS['green_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLORS['green_bright'])
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS['text_normal'])
    
    # Stop/Danger button
    with dpg.theme() as themes['stop']:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (*COLORS['orange_dim'], 200))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, COLORS['orange_normal'])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, COLORS['error'])
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS['text_normal'])
    
    # Disabled button
    with dpg.theme() as themes['disabled']:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (*COLORS['bg_light'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (*COLORS['bg_light'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (*COLORS['bg_light'], 100))
            dpg.add_theme_color(dpg.mvThemeCol_Text, COLORS['text_dim'])
    
    return themes