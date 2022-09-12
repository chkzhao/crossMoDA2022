class UNet2d5_spvPA(nn.Module):
    def __init__(
        self,
        dimensions,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_sizes,
        sample_kernel_sizes,
        num_res_units=0,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0,
        attention_module=True,
        zoom_in=False
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == (len(strides)) + 1 == len(sample_kernel_sizes) + 1
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.sample_kernel_sizes = sample_kernel_sizes
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.attention_module = attention_module
        self.att_maps = []
        self.zoom_in = zoom_in

        self.upsample1 = nn.Upsample(scale_factor=(1.5, 1.5, 1), mode='trilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        self.upsample3 = nn.Upsample(scale_factor=(3, 3, 1), mode='trilinear', align_corners=True)

        def _create_block(inc, outc, channels, strides, kernel_sizes, sample_kernel_sizes, is_top):
            """
            Builds the UNet2d5_spvPA structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]
            k = kernel_sizes[0]
            sk = sample_kernel_sizes[0]

            # create layer in downsampling path
            down = Down_Net(in_channels=inc, out_channels=c, kernel_size=k, num_res_units = self.num_res_units,
                            act=self.act,norm=self.norm,dropout=self.dropout,dimensions=self.dimensions)
            downsample = Down_sample(dimensions=self.dimensions, in_channels=c, out_channels=c, strides=s, kernel_size=sk, act = self.act,norm=self.norm,dropout=self.dropout)

            if len(channels) > 2:
                # continue recursion down
                subblock = _create_block(
                    c, channels[1], channels[1:], strides[1:], kernel_sizes[1:], sample_kernel_sizes[1:], is_top=False
                )
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(
                    in_channels=c,
                    out_channels=channels[1],
                    kernel_size=kernel_sizes[1],
                )

            upsample = self._get_upsample_layer(in_channels=channels[1], out_channels=c, strides=s, up_kernel_size=sk)
            subblock_with_resampling = nn.Sequential(downsample, subblock, upsample)

            # create layer in upsampling path
            up = self._get_up_layer(in_channels=2 * c, out_channels=outc, kernel_size=k, is_top=is_top)

            return nn.Sequential(down, SkipConnection(subblock_with_resampling), up)

        self.model = _create_block(
            in_channels, out_channels, self.channels, self.strides, self.kernel_sizes, self.sample_kernel_sizes, True
        )

        # register forward hooks on all Attentionblock1 modules, to save the attention maps
        if self.attention_module:
            for layer_id, layer in enumerate(self.model.modules()):
                if type(layer) == AttentionBlock1:
                    layer.register_forward_hook(self.hook_save_attention_map)

    def hook_save_attention_map(self, module, inp, outp):
        if len(self.att_maps) == len(self.channels):

            self.att_maps = []
        self.att_maps.append(outp[0])  # get first element of output (Attentionblock1 returns (att, x) )

    def _get_att_layer(self, in_channels, out_channels, kernel_size):
        att1 = AttentionBlock1(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        att2 = AttentionBlock2(self.dimensions, in_channels, out_channels, kernel_size, norm=None, dropout=self.dropout)

        return nn.Sequential(att1, att2)

    def _get_down_layer(self, in_channels, out_channels, kernel_size):
        if self.num_res_units > 0:
            return ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )
        else:
            return Convolution(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
            )

    def _get_bottom_layer(self, in_channels, out_channels, kernel_size):
        conv = self._get_down_layer(in_channels, out_channels, kernel_size)
        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)
            return nn.Sequential(att_layer, conv)
        else:
            return conv

    def _get_upsample_layer(self, in_channels, out_channels, strides, up_kernel_size):
        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides,
            up_kernel_size,
            self.act,
            self.norm,
            self.dropout,
            is_transposed=True,
        )
        return conv

    def _get_up_layer(self, in_channels, out_channels, kernel_size, is_top):

        if self.attention_module:
            att_layer = self._get_att_layer(in_channels, in_channels, kernel_size)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,  # why not self.num_res_units?
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                last_conv_only=is_top,
            )

        if self.attention_module and self.num_res_units > 0:
            return nn.Sequential(att_layer, ru)
        elif self.attention_module and not self.num_res_units > 0:
            return att_layer
        elif self.num_res_units > 0 and not self.attention_module:
            return ru
        elif not self.attention_module and not self.num_res_units > 0:
            return nn.Identity
        else:
            raise NotImplementedError

    def forward(self, input):
        if self.zoom_in:
            zoom_input1 = self.upsample1(input)
            zoom_input1 = center_crop(zoom_input1, size=[input.shape[2], input.shape[3], input.shape[4]])
            zoom_input2 = self.upsample2(input)
            zoom_input2 = center_crop(zoom_input2, size=[input.shape[2], input.shape[3], input.shape[4]])
            zoom_input3 = self.upsample3(input)
            zoom_input3 = center_crop(zoom_input3, size=[input.shape[2], input.shape[3], input.shape[4]])
            input = torch.cat([input, zoom_input1, zoom_input2, zoom_input3], dim=1)
        x = self.model(input)
        return x, self.att_maps