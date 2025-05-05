class E4M3Format:
    WIDTH = 8
    EXP_BITS = 4
    MAN_BITS = 3
    EXP_BIAS = 7
    NAN = 0x7F
    ZERO = 0x00

    @staticmethod
    def extract_components_constants():
        """Return constants needed for component extraction"""
        sign_mask = 1 << (E4M3Format.WIDTH - 1)
        exp_mask = ((1 << E4M3Format.EXP_BITS) - 1) << E4M3Format.MAN_BITS
        man_mask = (1 << E4M3Format.MAN_BITS) - 1

        return sign_mask, exp_mask, man_mask, E4M3Format.MAN_BITS
