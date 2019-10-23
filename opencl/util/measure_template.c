struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);


	init_time = probe_time_end(&ts_init);
    probe_time_start(&ts_memalloc);

    mem_alloc_time = probe_time_end(&ts_memalloc);
    probe_time_start(&ts_h2d);
    h2d_time = probe_time_end(&ts_h2d);
    probe_time_start(&ts_kernel);

    clFinish(cmd_queue);
    kernel_time = probe_time_end(&ts_kernel);
    probe_time_start(&ts_d2h);

    d2h_time += probe_time_end(&ts_d2h);

    probe_time_start(&ts_close);
	close_time = probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("HtoD_Phy: %f\n", h2d_phy_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("DtoH_Phy: %f\n", d2h_phy_time);
	printf("Close: %f\n", close_time);
	printf("API: %f\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	printf("Total: %f\n", total_time);

