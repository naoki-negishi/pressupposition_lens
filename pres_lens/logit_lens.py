# wrapper around running the op in the session

import json
import os

import encoder  # from gpt-2
import model  # from gpt-2
import tensorflow as tf

batch_size = 1


def internal_token_dists(
    hparams,
    X,
    layer_nums=None,
    use_normed_layers=True,
    use_affine_transformed_layers=True,
    max_tokens_to_return=hparams.n_ctx,
    truncate_at_right=True,
    only_first_batch_element=True,
    return_logits=True,
    return_probs=True,
    return_argmaxes=True,
    return_positions=False,
    return_activations=False,
    past=None,
    past_select=None,
    scope="model",
    reuse=tf.AUTO_REUSE,
):
    layer_logits = []
    layer_positions = []
    actis = []
    h_names = []

    dtype = tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        if truncate_at_right:
            X = X[:, :max_tokens_to_return]
        batch, sequence = shape_list(X)

        # positions?
        wpe = tf.get_variable(
            "wpe",
            [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype),
        )
        # embeddings? - word embeddings
        wte = tf.get_variable(
            "wte",
            [hparams.n_vocab, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype),
        )
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        def _get_logits_from_h(h_):
            h_flat = tf.reshape(h_, [batch * sequence, hparams.n_embd])
            logits = tf.matmul(h_flat, wte, transpose_b=True)
            logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
            if truncate_at_right:
                logits = logits[:, :max_tokens_to_return, :]
            else:
                logits = logits[:, -max_tokens_to_return:, :]
            return logits

        def _get_positions_from_h(h_):
            h_flat = tf.reshape(h_, [batch * sequence, hparams.n_embd])
            positions = tf.matmul(h_flat, wpe, transpose_b=True)
            positions = tf.reshape(positions, [batch, sequence, hparams.n_ctx])
            if truncate_at_right:
                positions = positions[:, :max_tokens_to_return, :]
            else:
                positions = positions[:, -max_tokens_to_return:, :]
            return positions

        h_names.append("h_in")
        if use_normed_layers:
            actis.append(fixed_norm(h, "ln_fixed_in", hparams=hparams))
        else:
            actis.append(h)
        layer_logits.append(_get_logits_from_h(actis[-1]))
        layer_positions.append(_get_positions_from_h(actis[-1]))

        def norm_returning_block(x, scope, *, past, hparams):
            dtype = tf.float32
            with tf.variable_scope(scope, dtype=dtype):
                nx = x.shape[-1].value
                norm_in = norm(x, 'ln_1',)
                a, present = attn(norm_in, 'attn', nx, past=past, hparams=hparams)
                x = x + a
                ln_2 = norm(x, 'ln_2',)
                m = mlp(ln_2, 'mlp', nx*4, hparams=hparams)
                x = x + m
                return x, present, norm_in

        # Transformer
        presents = []
        pasts = (
            tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        )
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            # これがtransformerの計算の本体かな？
            h, present, norm_in = norm_returning_block(
                h, "h%d" % layer, past=past, hparams=hparams
            )
            presents.append(present)

            if layer_nums is None or layer in layer_nums:
                if use_normed_layers and use_affine_transformed_layers:
                    actis.append(norm_in)
                    h_name = f"h{layer}_in"
                elif use_normed_layers:
                    actis.append(fixed_norm(h, f"ln_fixed_in_{layer}", hparams=hparams))
                    h_name = f"h{layer}_out"
                else:
                    actis.append(h)
                    h_name = f"h{layer}_out"
                layer_logits.append(_get_logits_from_h(actis[-1]))
                layer_positions.append(_get_positions_from_h(actis[-1]))
                h_names.append(h_name)

        h = norm(h, "ln_f")

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])

        h_names.append("h_out")
        actis.append(h)
        layer_logits.append(logits)
        layer_positions.append(_get_positions_from_h(actis[-1]))

        results = defaultdict(list)
        for l, p, a in zip(layer_logits, layer_positions, actis):
            if return_logits:
                results["logits"].append(l)
            if return_probs:
                results["probs"].append(tf.nn.softmax(l, axis=-1))
            if return_argmaxes:
                results["argmaxes"].append(tf.argmax(l, axis=-1))
            if return_positions:
                results["positions"].append(p)
            if return_activations:
                if truncate_at_right:
                    results["acti"].append(a[:, :max_tokens_to_return, :])
                else:
                    results["acti"].append(a[:, -max_tokens_to_return:, :])

        if only_first_batch_element:
            for name in results.keys():
                results[name] = [entry[0, ...] for entry in results[name]]

        layer_names = tf.constant(
            [h_names], shape=(batch_size, len(h_names)), dtype=tf.string
        )
        if only_first_batch_element:
            layer_names = layer_names[0, ...]

        return results, layer_names


def load_model(model_name: str):
    model_name = "1558M"
    enc = encoder.get_encoder(model_name, "models/")
    hparams = model.default_hparams()

    with open(os.path.join(f"models/{model_name}/hparams.json")) as f:
        hparam_dict = json.load(f)

    hparams.override_from_dict(hparam_dict)  # tensorflow method


def run_example(text):
    with sess.as_default():
        context = tf.placeholder(tf.int32, [batch_size, None])
        get_internal_token_dists_200_leftmost = internal_token_dists(
            hparams=hparams,
            X=context,
            use_affine_transformed_layers=False,
            use_normed_layers=True,
            max_tokens_to_return=200,
            truncate_at_right=True,
        )

    tokens = enc.encode(text)
    my_context = [tokens for _ in range(batch_size)]
    with sess.as_default():
        results, layer_names = sess.run(
            get_internal_token_dists_200_leftmost, feed_dict={context.name: my_context}
        )

    layer_names = np.asarray([b.decode() for b in layer_names])  # bytes to str

    return tokens[:200], results, layer_names


def main():
    try:
        sess.close()
    except:
        pass

    batch_size = 1
    tf.reset_default_graph()

    sess = tf.Session()
