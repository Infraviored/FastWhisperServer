#!/bin/bash

is_terminal() {
    WINDOW_CLASS=$(xprop -id $(xdotool getactivewindow) WM_CLASS | cut -d'"' -f2)
    case "$WINDOW_CLASS" in
        "gnome-terminal-server"|"gnome-terminal"|"x-terminal-emulator"|"terminator"|"xfce4-terminal"|"konsole"|"kitty"|"alacritty"|"urxvt"|"rxvt"|"xterm"|"tilix"|"terminology"|"sakura"|"guake"|"tilda"|"st-256color"|"st"|"roxterm"|"lxterminal"|"qt5term"|"qterminal"|"mate-terminal"|"pantheon-terminal"|"finalterm"|"cool-retro-term"|"deepin-terminal")
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if is_terminal; then
    xdotool keydown ctrl keydown shift key v keyup shift keyup ctrl
else
    xdotool keydown ctrl key v keyup ctrl
fi
