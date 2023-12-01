import threading

from modules.params import HandlerParams, UovMethod, OutpaintDirection, CnTask, AspectRatio


class AsyncTask:
    def __init__(self, args):
        self.args = args
        self.yields = []
        self.results = []


async_tasks = []


def worker():
    global async_tasks

    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import fcbh.model_management
    import fooocus_extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import modules.advanced_parameters as advanced_parameters
    import fooocus_extras.ip_adapter as ip_adapter
    import fooocus_extras.face_crop

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image
    from modules.upscaler import perform_upscale

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        async_task.yields.append(['preview', (number, text, None)])

    def yield_result(async_task, imgs, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return

    def build_image_wall(async_task):
        if not advanced_parameters.generate_image_grid:
            return

        results = async_task.results

        if len(results) < 2:
            return

        for img in results:
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    def web_handler(async_task):
        params: HandlerParams = HandlerParams()
        args: list = async_task.args
        args.reverse()

        params.prompt = args.pop()
        params.negative_prompt = args.pop()
        params.style_selections = args.pop()
        params.steps = args.pop()
        params.aspect_ratio = AspectRatio.from_str(args.pop())
        params.im_count = int(args.pop())
        params.im_seed = int(args.pop())
        params.sharpness = float(args.pop())
        params.guidance_scale = float(args.pop())
        params.base_model_name = args.pop()
        params.refiner_model_name = args.pop()
        params.refiner_switch = args.pop()
        params.loras = [[str(args.pop()), float(args.pop())] for _ in range(5)]
        params.input_image_checkbox = args.pop()
        params.current_tab = args.pop()
        params.uov_method = UovMethod.from_str(args.pop())
        params.uov_input_image = args.pop()
        params.outpaint_directions = OutpaintDirection.from_str(args.pop())
        params.inpaint_input_image = args.pop()
        params.inpaint_additional_prompt = args.pop()

        for _ in range(4):
            cn_task = CnTask()
            cn_task.im = args.pop()
            cn_task.stop = args.pop()
            cn_task.weight = args.pop()
            cn_type = args.pop()
            if cn_task.im is not None:
                match cn_type:
                    case flags.cn_ip:
                        params.cn_image_prompt.append(cn_task)
                    case flags.cn_ip_face:
                        params.cn_face_swap.append(cn_task)
                    case flags.cn_canny:
                        params.cn_pyra_canny.append(cn_task)
                    case flags.cn_cpds:
                        params.cn_cpds.append(cn_task)

        if params.steps == "Speed":
            params.steps = 30
        elif params.steps == "Quality":
            params.steps = 60
        elif params.steps == "Extreme Speed":
            params.steps = 8

        async_task.args = params

        handler(async_task)

    def py_handler(handler_params: HandlerParams):
        ...  # TODO

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task):
        execution_start_time = time.perf_counter()

        args: HandlerParams = async_task.args

        base_model_additional_loras = []
        style_selections = [style for style in args.style_selections]
        raw_style_selections = copy.deepcopy(style_selections)

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if args.base_model_name == args.refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            args.refiner_model_name = None

        if args.steps < 10:  # extreme speed
            print('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            args.loras.append(modules.config.downloading_sdxl_lcm_lora(), 1.0)

            if args.refiner_model_name is not None:
                print(f'Refiner disabled in LCM mode.')

            args.refiner_model_name = None
            modules.patch.sharpness = args.sharpness = 0.0
            guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            refiner_switch = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            args.steps = 8

        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = args.sharpness
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        print(f'[Parameters] ADM Scale = '
              f'{modules.patch.positive_adm_scale} : '
              f'{modules.patch.negative_adm_scale} : '
              f'{modules.patch.adm_scaler_end}')

        cfg_scale = float(args.guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = args.aspect_ratio.value.split('x')
        width, height = int(width), int(height)

        skip_prompt_processing = False
        refiner_swap_method = advanced_parameters.refiner_swap_method

        inpaint_worker.current_task = None
        inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        print(f'[Parameters] Seed = {args.im_seed}')

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if args.input_image_checkbox:
            if (args.current_tab == 'uov' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and args.uov_method.value != flags.disabled and args.uov_input_image is not None:
                uov_input_image = HWC3(args.uov_input_image)
                if args.uov_method == UovMethod.VARY:
                    goals.append('vary')
                elif args.uov_method in [UovMethod.UPSCALE_1_5, UovMethod.UPSCALE_2, UovMethod.UPSCALE]:
                    goals.append('upscale')
                    if args.uov_method == UovMethod.FAST:
                        skip_prompt_processing = True
                    else:
                        args.steps = 18
                        if args.steps == 30:
                            args.steps = 18
                        elif args.steps == 60:
                            args.steps = 36

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (args.current_tab == 'inpaint' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint)) \
                    and isinstance(args.inpaint_input_image, dict):
                inpaint_image = args.inpaint_input_image['image']
                inpaint_mask = args.inpaint_input_image['mask'][:, :, 0]
                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(args.outpaint_directions) > 0):
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        modules.config.downloading_upscale_model()
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            advanced_parameters.inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if args.refiner_model_name is None:
                            use_synthetic_refiner = True
                            args.refiner_switch = 0.5
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if args.inpaint_additional_prompt != '':
                        if args.prompt == '':
                            args.prompt = args.inpaint_additional_prompt
                        else:
                            args.prompt = args.inpaint_additional_prompt + '\n' + args.prompt
                    goals.append('inpaint')
            if args.current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(args.cn_pyra_canny) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(args.cn_cpds) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(args.cn_image_prompt) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(args.cn_face_swap) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        switch = int(round(args.steps * args.refiner_switch))

        if advanced_parameters.overwrite_step > 0:
            args.steps = advanced_parameters.overwrite_step

        if advanced_parameters.overwrite_switch > 0:
            switch = advanced_parameters.overwrite_switch

        if advanced_parameters.overwrite_width > 0:
            width = advanced_parameters.overwrite_width

        if advanced_parameters.overwrite_height > 0:
            height = advanced_parameters.overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {args.steps} - {switch}')

        progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in args.prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in args.negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=args.refiner_model_name,
                                        base_model_name=args.base_model_name,
                                        loras=args.loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            for i in range(args.im_count):
                task_seed = (args.im_seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

                task_prompt = apply_wildcards(prompt, task_rng)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='; '.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='; '.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in goals:
            if args.uov_method == UovMethod.SUBTLE:
                denoising_strength = 0.5
            if args.uov_method == UovMethod.STRONG:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(args.uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(args.uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=args.steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = args.uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(args.uov_input_image)
            print(f'Image upscaled.')

            if args.uov_method == UovMethod.UPSCALE_1_5:
                f = 1.5
            elif args.uov_method == UovMethod.UPSCALE_2:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if args.uov_method == UovMethod.FAST:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', '2x')]
                log(uov_input_image, d, single_line_number=1)
                yield_result(async_task, uov_input_image, do_not_show_finished_images=True)
                return

            tiled = True
            denoising_strength = 0.382

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=args.steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(args.outpaint_directions) > 0:
                H, W, C = inpaint_image.shape
                if OutpaintDirection.TOP in args.outpaint_directions:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if OutpaintDirection.BOTTOM in args.outpaint_directions:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if OutpaintDirection.LEFT in args.outpaint_directions:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if OutpaintDirection.RIGHT in args.outpaint_directions:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                advanced_parameters.inpaint_strength = 1.0
                advanced_parameters.inpaint_respective_field = 1.0

            denoising_strength = advanced_parameters.inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=advanced_parameters.inpaint_respective_field
            )

            if advanced_parameters.debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                             do_not_show_finished_images=True)
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=args.steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not advanced_parameters.inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in args.cn_pyra_canny:
                task.im = resize_image(HWC3(task.im), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    task.im = preprocessors.canny_pyramid(task.im)

                task.im = HWC3(task.im)
                task[0] = core.numpy_to_pytorch(task.im)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, task.im, do_not_show_finished_images=True)
                    return
            for task in args.cn_cpds:
                task.im = resize_image(HWC3(task.im), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    task.im = preprocessors.cpds(task.im)

                task.im = HWC3(task.im)
                task.im = core.numpy_to_pytorch(task.im)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, task.im, do_not_show_finished_images=True)
                    return
            for task in args.cn_image_prompt:
                task.im = HWC3(task.im)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                task.im = resize_image(task.im, width=224, height=224, resize_mode=0)

                task.im = ip_adapter.preprocess(task.im, ip_adapter_path=ip_adapter_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, task.im, do_not_show_finished_images=True)
                    return
            for task in args.cn_face_swap:
                task.im = HWC3(task.im)

                if not advanced_parameters.skipping_cn_preprocessor:
                    task.im = fooocus_extras.face_crop.crop_image(task.im)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                task.im = resize_image(task.im, width=224, height=224, resize_mode=0)

                task.im = ip_adapter.preprocess(task.im, ip_adapter_path=ip_adapter_face_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, task.im, do_not_show_finished_images=True)
                    return

            all_ip_tasks = args.cn_image_prompt + args.cn_face_swap

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if advanced_parameters.freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = args.steps * args.im_count

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * args.steps + step
            async_task.yields.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        cn_tasks = {
            flags.cn_ip: args.cn_image_prompt,
            flags.cn_ip_face: args.cn_face_swap,
            flags.cn_canny: args.cn_pyra_canny,
            flags.cn_cpds: args.cn_cpds
        }
        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
                    steps=args.steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                for x in imgs:
                    d = [
                        ('Prompt', task['log_positive_prompt']),
                        ('Negative Prompt', task['log_negative_prompt']),
                        ('Fooocus V2 Expansion', task['expansion']),
                        ('Styles', str(raw_style_selections)),
                        ('Performance', args.steps),
                        ('Resolution', str((width, height))),
                        ('Sharpness', args.sharpness),
                        ('Guidance Scale', args.guidance_scale),
                        ('ADM Guidance', str((
                            modules.patch.positive_adm_scale,
                            modules.patch.negative_adm_scale,
                            modules.patch.adm_scaler_end))),
                        ('Base Model', args.base_model_name),
                        ('Refiner Model', args.refiner_model_name),
                        ('Refiner Switch', args.refiner_switch),
                        ('Sampler', sampler_name),
                        ('Scheduler', scheduler_name),
                        ('Seed', task['task_seed'])
                    ]
                    for n, w in args.loras:
                        if n != 'None':
                            d.append((f'LoRA [{n}] weight', w))
                    log(x, d, single_line_number=3)

                yield_result(async_task, imgs, do_not_show_finished_images=len(tasks) == 1)
            except fcbh.model_management.InterruptProcessingException as e:
                if shared.last_stop == 'skip':
                    print('User skipped')
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')

        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            try:
                web_handler(task)
            except:
                traceback.print_exc()
            finally:
                build_image_wall(task)
                task.yields.append(['finish', task.results])
                pipeline.prepare_text_encoder(async_call=True)
    pass


threading.Thread(target=worker, daemon=True).start()
