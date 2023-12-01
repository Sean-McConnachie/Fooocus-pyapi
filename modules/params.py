from enum import Enum

import numpy as np


class HandlerParams:
    prompt: str
    negative_prompt: str
    style_selections: list[str]  # SdxlStyles
    steps: int
    aspect_ratio: "AspectRatio"

    im_count: int
    im_seed: int
    sharpness: float
    guidance_scale: float

    base_model_name: str
    refiner_model_name: str | None
    refiner_switch: float
    loras: list[tuple[str | None, float]]
    input_image_checkbox: bool
    current_tab: str

    uov_method: "UovMethod"
    uov_input_image: np.ndarray | None

    outpaint_directions: list["OutpaintDirection"]
    inpaint_input_image: dict[str, np.ndarray] | None
    inpaint_additional_prompt: str

    cn_image_prompt: list["CnTask"]
    cn_face_swap: list["CnTask"]
    cn_pyra_canny: list["CnTask"]
    cn_cpds: list["CnTask"]

    def __init__(self):
        self.cn_image_prompt = []
        self.cn_face_swap = []
        self.cn_pyra_canny = []
        self.cn_cpds = []


class AspectRatio(Enum):
    A1b2 = "704x1408"
    A11b21 = "704x1344"
    A4b7 = "768x1344"
    A3b5 = "768x1280"
    A13b19 = "832x1216"
    A13b18 = "832x1152"
    A7b9 = "896x1152"
    A14b17 = "896x1088"
    A15b17 = "960x1088"
    A15b16 = "960x1024"
    A1b1 = "1024x1024"
    A16b15 = "1024x960"
    A17b15 = "1088x960"
    A17b14 = "1088x896"
    A9b7 = "1152x896"
    A18b13 = "1152x832"
    A19b13 = "1216x832"
    A5b3 = "1280x768"
    A7b4 = "1344x768"
    A21b11 = "1344x704"
    A2b1 = "1408x704"
    A23b11 = "1472x704"
    A12b5 = "1536x640"
    A5b2 = "1600x640"
    A26b9 = "1664x576"
    A3b1 = "1728x576"

    @staticmethod
    def from_str(s: str) -> "AspectRatio":
        width, height = s.replace('×', ' ').split(' ')[:2]
        for v in AspectRatio.__members__.values():
            if v.value == f"{width}x{height}":
                return v


class UovMethod(Enum):
    DISABLED = "Disabled"
    ENABLED = "Enabled"
    VARY = "vary"
    UPSCALE = "upscale"
    FAST = "fast"
    SUBTLE = "subtle"
    STRONG = "strong"
    UPSCALE_1_5 = "1.5x"
    UPSCALE_2 = "2x"

    @staticmethod
    def from_str(s: str) -> list["UovMethod"]:
        for v in s.split(' '):
            if v in UovMethod.__members__:
                yield UovMethod[v]


class OutpaintDirection(Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

    @staticmethod
    def from_str(s: str) -> list["OutpaintDirection"]:
        for v in s.split(' '):
            if v in OutpaintDirection.__members__:
                yield OutpaintDirection[v]


class CnTask:
    im: np.ndarray
    stop: float
    weight: float

    def __init__(self):
        self.im = None  # noqa
        self.stop = None  # noqa
        self.weight = None  # noqa

    def __iter__(self):
        yield self.im
        yield self.stop
        yield self.weight

    def __setitem__(self, key, value):
        if key == 0:
            self.im = value
        elif key == 1:
            self.stop = value
        elif key == 2:
            self.weight = value
        else:
            raise IndexError


class SdxlStyles(Enum):
    ...


class SdxlStylesDiva(SdxlStyles):
    CINEMATIC_DIVA = "cinematic-diva"
    ABSTRACT_EXPRESSIONISM = "Abstract Expressionism"
    ACADEMIA = "Academia"
    ACTION_FIGURE = "Action Figure"
    ADORABLE_3D_CHARACTER = "Adorable 3D Character"
    ADORABLE_KAWAII = "Adorable Kawaii"
    ART_DECO = "Art Deco"
    ART_NOUVEAU = "Art Nouveau"
    ASTRAL_AURA = "Astral Aura"
    AVANT_GARDE = "Avant-garde"
    BAROQUE = "Baroque"
    BAUHAUS_STYLE_POSTER = "Bauhaus-Style Poster"
    BLUEPRINT_SCHEMATIC_DRAWING = "Blueprint Schematic Drawing"
    CARICATURE = "Caricature"
    CEL_SHADED_ART = "Cel Shaded Art"
    CHARACTER_DESIGN_SHEET = "Character Design Sheet"
    CLASSICISM_ART = "Classicism Art"
    COLOR_FIELD_PAINTING = "Color Field Painting"
    COLORED_PENCIL_ART = "Colored Pencil Art"
    CONCEPTUAL_ART = "Conceptual Art"
    CONSTRUCTIVISM = "Constructivism"
    CUBISM = "Cubism"
    DADAISM = "Dadaism"
    DARK_FANTASY = "Dark Fantasy"
    DARK_MOODY_ATMOSPHERE = "Dark Moody Atmosphere"
    DMT_ART_STYLE = "DMT Art Style"
    DOODLE_ART = "Doodle Art"
    DOUBLE_EXPOSURE = "Double Exposure"
    DRIPPING_PAINT_SPLATTER_ART = "Dripping Paint Splatter Art"
    EXPRESSIONISM = "Expressionism"
    FADED_POLAROID_PHOTO = "Faded Polaroid Photo"
    FAUVISM = "Fauvism"
    FLAT_2D_ART = "Flat 2D Art"
    FORTNITE_ART_STYLE = "Fortnite Art Style"
    FUTURISM = "Futurism"
    GLITCHCORE = "Glitchcore"
    GLO_FI = "Glo-fi"
    GOOGIE_ART_STYLE = "Googie Art Style"
    GRAFFITI_ART = "Graffiti Art"
    HARLEM_RENAISSANCE_ART = "Harlem Renaissance Art"
    HIGH_FASHION = "High Fashion"
    IDYLLIC = "Idyllic"
    IMPRESSIONISM = "Impressionism"
    INFOGRAPHIC_DRAWING = "Infographic Drawing"
    INK_DRIPPING_DRAWING = "Ink Dripping Drawing"
    JAPANESE_INK_DRAWING = "Japanese Ink Drawing"
    KNOLLING_PHOTOGRAPHY = "Knolling Photography"
    LIGHT_CHEERY_ATMOSPHERE = "Light Cheery Atmosphere"
    LOGO_DESIGN = "Logo Design"
    LUXURIOUS_ELEGANCE = "Luxurious Elegance"
    MACRO_PHOTOGRAPHY = "Macro Photography"
    MANDOLA_ART = "Mandola Art"
    MARKER_DRAWING = "Marker Drawing"
    MEDIEVALISM = "Medievalism"
    MINIMALISM = "Minimalism"
    NEO_BAROQUE = "Neo-Baroque"
    NEO_BYZANTINE = "Neo-Byzantine"
    NEO_FUTURISM = "Neo-Futurism"
    NEO_IMPRESSIONISM = "Neo-Impressionism"
    NEO_ROCOCO = "Neo-Rococo"
    NEOCLASSICISM = "Neoclassicism"
    OP_ART = "Op Art"
    ORNATE_AND_INTRICATE = "Ornate and Intricate"
    PENCIL_SKETCH_DRAWING = "Pencil Sketch Drawing"
    POP_ART_2 = "Pop Art 2"
    ROCOCO = "Rococo"
    SILHOUETTE_ART = "Silhouette Art"
    SIMPLE_VECTOR_ART = "Simple Vector Art"
    SKETCHUP = "Sketchup"
    STEAMPUNK_2 = "Steampunk 2"
    SURREALISM = "Surrealism"
    SUPREMATISM = "Suprematism"
    TERRAGEN = "Terragen"
    TRANQUIL_RELAXING_ATMOSPHERE = "Tranquil Relaxing Atmosphere"
    STICKER_DESIGNS = "Sticker Designs"
    VIBRANT_RIM_LIGHT = "Vibrant Rim Light"
    VOLUMETRIC_LIGHTING = "Volumetric Lighting"
    WATERCOLOR_2 = "Watercolor 2"
    WHIMSICAL_AND_PLAYFUL = "Whimsical and Playful"


class SdxlStylesFooocus(SdxlStyles):
    FOOOCUS_ENHANCE = "Fooocus Enhance"
    FOOOCUS_SHARP = "Fooocus Sharp"
    FOOOCUS_MASTERPIECE = "Fooocus Masterpiece"
    FOOOCUS_PHOTOGRAPH = "Fooocus Photograph"
    FOOOCUS_NEGATIVE = "Fooocus Negative"
    FOOOCUS_CINEMATIC = "Fooocus Cinematic"


class SdxlStylesMre(SdxlStyles):
    MRE_CINEMATIC_DYNAMIC = "mre-cinematic-dynamic"
    MRE_SPONTANEOUS_PICTURE = "mre-spontaneous-picture"
    MRE_ARTISTIC_VISION = "mre-artistic-vision"
    MRE_DARK_DREAM = "mre-dark-dream"
    MRE_GLOOMY_ART = "mre-gloomy-art"
    MRE_BAD_DREAM = "mre-bad-dream"
    MRE_UNDERGROUND = "mre-underground"
    MRE_SURREAL_PAINTING = "mre-surreal-painting"
    MRE_DYNAMIC_ILLUSTRATION = "mre-dynamic-illustration"
    MRE_UNDEAD_ART = "mre-undead-art"
    MRE_ELEMENTAL_ART = "mre-elemental-art"
    MRE_SPACE_ART = "mre-space-art"
    MRE_ANCIENT_ILLUSTRATION = "mre-ancient-illustration"
    MRE_BRAVE_ART = "mre-brave-art"
    MRE_HEROIC_FANTASY = "mre-heroic-fantasy"
    MRE_DARK_CYBERPUNK = "mre-dark-cyberpunk"
    MRE_LYRICAL_GEOMETRY = "mre-lyrical-geometry"
    MRE_SUMI_E_SYMBOLIC = "mre-sumi-e-symbolic"
    MRE_SUMI_E_DETAILED = "mre-sumi-e-detailed"
    MRE_MANGA = "mre-manga"
    MRE_ANIME = "mre-anime"
    MRE_COMIC = "mre-comic"


class SdxlStylesSai(SdxlStyles):
    SAI_3D_MODEL = "sai-3d-model"
    SAI_ANALOG_FILM = "sai-analog-film"
    SAI_ANIME = "sai-anime"
    SAI_CINEMATIC = "sai-cinematic"
    SAI_COMIC_BOOK = "sai-comic-book"
    SAI_CRAFT_CLAY = "sai-craft-clay"
    SAI_DIGITAL_ART = "sai-digital-art"
    SAI_ENHANCE = "sai-enhance"
    SAI_FANTASY_ART = "sai-fantasy-art"
    SAI_ISOMETRIC = "sai-isometric"
    SAI_LINE_ART = "sai-line-art"
    SAI_LOWPOLY = "sai-lowpoly"
    SAI_NEONPUNK = "sai-neonpunk"
    SAI_ORIGAMI = "sai-origami"
    SAI_PHOTOGRAPHIC = "sai-photographic"
    SAI_PIXEL_ART = "sai-pixel-art"
    SAI_TEXTURE = "sai-texture"


class SdxlStylesTwri(SdxlStyles):
    ADS_ADVERTISING = "ads-advertising"
    ADS_AUTOMOTIVE = "ads-automotive"
    ADS_CORPORATE = "ads-corporate"
    ADS_FASHION_EDITORIAL = "ads-fashion editorial"
    ADS_FOOD_PHOTOGRAPHY = "ads-food photography"
    ADS_GOURMET_FOOD_PHOTOGRAPHY = "ads-gourmet food photography"
    ADS_LUXURY = "ads-luxury"
    ADS_REAL_ESTATE = "ads-real estate"
    ADS_RETAIL = "ads-retail"
    ARTSTYLE_ABSTRACT = "artstyle-abstract"
    ARTSTYLE_ABSTRACT_EXPRESSIONISM = "artstyle-abstract expressionism"
    ARTSTYLE_ART_DECO = "artstyle-art deco"
    ARTSTYLE_ART_NOUVEAU = "artstyle-art nouveau"
    ARTSTYLE_CONSTRUCTIVIST = "artstyle-constructivist"
    ARTSTYLE_CUBIST = "artstyle-cubist"
    ARTSTYLE_EXPRESSIONIST = "artstyle-expressionist"
    ARTSTYLE_GRAFFITI = "artstyle-graffiti"
    ARTSTYLE_HYPERREALISM = "artstyle-hyperrealism"
    ARTSTYLE_IMPRESSIONIST = "artstyle-impressionist"
    ARTSTYLE_POINTILLISM = "artstyle-pointillism"
    ARTSTYLE_POP_ART = "artstyle-pop art"
    ARTSTYLE_PSYCHEDELIC = "artstyle-psychedelic"
    ARTSTYLE_RENAISSANCE = "artstyle-renaissance"
    ARTSTYLE_STEAMPUNK = "artstyle-steampunk"
    ARTSTYLE_SURREALIST = "artstyle-surrealist"
    ARTSTYLE_TYPOGRAPHY = "artstyle-typography"
    ARTSTYLE_WATERCOLOR = "artstyle-watercolor"
    FUTURISTIC_BIOMECHANICAL = "futuristic-biomechanical"
    FUTURISTIC_BIOMECHANICAL_CYBERPUNK = "futuristic-biomechanical cyberpunk"
    FUTURISTIC_CYBERNETIC = "futuristic-cybernetic"
    FUTURISTIC_CYBERNETIC_ROBOT = "futuristic-cybernetic robot"
    FUTURISTIC_CYBERPUNK_CITYSCAPE = "futuristic-cyberpunk cityscape"
    FUTURISTIC_FUTURISTIC = "futuristic-futuristic"
    FUTURISTIC_RETRO_CYBERPUNK = "futuristic-retro cyberpunk"
    FUTURISTIC_RETRO_FUTURISM = "futuristic-retro futurism"
    FUTURISTIC_SCI_FI = "futuristic-sci-fi"
    FUTURISTIC_VAPORWAVE = "futuristic-vaporwave"
    GAME_BUBBLE_BOBBLE = "game-bubble bobble"
    GAME_CYBERPUNK_GAME = "game-cyberpunk game"
    GAME_FIGHTING_GAME = "game-fighting game"
    GAME_GTA = "game-gta"
    GAME_MARIO = "game-mario"
    GAME_MINECRAFT = "game-minecraft"
    GAME_POKEMON = "game-pokemon"
    GAME_RETRO_ARCADE = "game-retro arcade"
    GAME_RETRO_GAME = "game-retro game"
    GAME_RPG_FANTASY_GAME = "game-rpg fantasy game"
    GAME_STRATEGY_GAME = "game-strategy game"
    GAME_STREETFIGHTER = "game-streetfighter"
    GAME_ZELDA = "game-zelda"
    MISC_ARCHITECTURAL = "misc-architectural"
    MISC_DISCO = "misc-disco"
    MISC_DREAMSCAPE = "misc-dreamscape"
    MISC_DYSTOPIAN = "misc-dystopian"
    MISC_FAIRY_TALE = "misc-fairy tale"
    MISC_GOTHIC = "misc-gothic"
    MISC_GRUNGE = "misc-grunge"
    MISC_HORROR = "misc-horror"
    MISC_KAWAII = "misc-kawaii"
    MISC_LOVECRAFTIAN = "misc-lovecraftian"
    MISC_MACABRE = "misc-macabre"
    MISC_MANGA = "misc-manga"
    MISC_METROPOLIS = "misc-metropolis"
    MISC_MINIMALIST = "misc-minimalist"
    MISC_MONOCHROME = "misc-monochrome"
    MISC_NAUTICAL = "misc-nautical"
    MISC_SPACE = "misc-space"
    MISC_STAINED_GLASS = "misc-stained glass"
    MISC_TECHWEAR_FASHION = "misc-techwear fashion"
    MISC_TRIBAL = "misc-tribal"
    MISC_ZENTANGLE = "misc-zentangle"
    PAPERCRAFT_COLLAGE = "papercraft-collage"
    PAPERCRAFT_FLAT_PAPERCUT = "papercraft-flat papercut"
    PAPERCRAFT_KIRIGAMI = "papercraft-kirigami"
    PAPERCRAFT_PAPER_MACHE = "papercraft-paper mache"
    PAPERCRAFT_PAPER_QUILLING = "papercraft-paper quilling"
    PAPERCRAFT_PAPERCUT_COLLAGE = "papercraft-papercut collage"
    PAPERCRAFT_PAPERCUT_SHADOW_BOX = "papercraft-papercut shadow box"
    PAPERCRAFT_STACKED_PAPERCUT = "papercraft-stacked papercut"
    PAPERCRAFT_THICK_LAYERED_PAPERCUT = "papercraft-thick layered papercut"
    PHOTO_ALIEN = "photo-alien"
    PHOTO_FILM_NOIR = "photo-film noir"
    PHOTO_GLAMOUR = "photo-glamour"
    PHOTO_HDR = "photo-hdr"
    PHOTO_IPHONE_PHOTOGRAPHIC = "photo-iphone photographic"
    PHOTO_LONG_EXPOSURE = "photo-long exposure"
    PHOTO_NEON_NOIR = "photo-neon noir"
    PHOTO_SILHOUETTE = "photo-silhouette"
    PHOTO_TILT_SHIFT = "photo-tilt-shift"
