"""SM100 GEMM launch and code-generation contract."""

from .config import persistent_gemm_settings


class Sm100TableGemm:
    provider_name = "sm100_tma"
    requires_table = True
    supports_streaming = False
    supports_mixed_tiles = True
    full_operand_descriptors = True
    direct_store_with_splits = True
    generic_import = ""

    @property
    def table_import(self):
        imports = (
            "from megabake.bodies.cuda.sm100.tcgen_gemm import (\n"
            "    Sm100TmaGemm,\n"
            "    build_sm100_table_descriptors as build_table_descriptors,\n"
            ")\n"
            "from megabake.bodies.cuda.sm100.persistent_gemm import Sm100PersistentGemm\n"
            "from megabake.bodies.cuda.sm100.descriptors import table_descriptors\n"
        )
        if persistent_gemm_settings().cutlass_fork:
            imports += "from megabake.bodies.cuda.sm100.cutlass_gemm import Sm100CutlassForkGemm\n"
        return imports

    def generic_definition(self, symbol, tile, cluster):
        raise ValueError("this GEMM body requires a cooperative phase-table launch")

    def table_definition(self, spec, tile, epilogues, streaming, stream_stages):
        settings = persistent_gemm_settings()
        persistent, width, stages, cutlass_fork = (
            settings.enabled,
            settings.width,
            settings.stages,
            settings.cutlass_fork,
        )
        symbol = (
            "Sm100TmaGemm"
            if persistent and width == 128
            else "Sm100CutlassForkGemm"
            if cutlass_fork
            else "Sm100PersistentGemm"
            if persistent
            else "Sm100TmaGemm"
        )
        return (
            f"{spec['symbol']} = {symbol}({tile[0]}, {spec['splits']}, {tile[1]}"
            + (f", acc_stages={stages}" if persistent and width == 256 else "")
            + (", warp_specialized=True" if persistent and width == 128 else "")
            + (f", epilogues=({epilogues})" if epilogues else "")
            + ")"
        )

    def source_object(self):
        settings = persistent_gemm_settings()
        if settings.enabled:
            if settings.cutlass_fork:
                from .cutlass_gemm import PersistentDenseGemmKernel

                return PersistentDenseGemmKernel.table_kernel
            if settings.width == 128:
                from .tcgen_gemm import _run_tcgen_tile

                return _run_tcgen_tile
            from .persistent_gemm import _persistent_run

            return _persistent_run
        from .tcgen_gemm import _run_tcgen_tile

        return _run_tcgen_tile


TABLE_GEMM = Sm100TableGemm()
