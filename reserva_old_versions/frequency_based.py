
from nltk.corpus import stopwords
#from .stop_words import ENGLISH_STOP_WORDS_17
import re

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted, check_array, FLOAT_DTYPES

replace_patterns = [
    ('<[^>]*>', ''),                                    # remove HTML tags
    ('(\D)\d\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d:\d\d(\D)', '\\1 ParsedTime \\2'),
    ('(\D)\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D\D)\d\d\d\D\D\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedPhoneNum \\2'),
    ('(\D)\d\d\d\d\d\-\d\d\d\d(\D)', '\\1 ParsedZipcodePlusFour \\2'),
    ('(\D)\d(\D)', '\\1ParsedOneDigit\\2'),
    ('(\D)\d\d(\D)', '\\1ParsedTwoDigits\\2'),
    ('(\D)\d\d\d(\D)', '\\1ParsedThreeDigits\\2'),
    ('(\D)\d\d\d\d(\D)', '\\1ParsedFourDigits\\2'),
    ('(\D)\d\d\d\d\d(\D)', '\\1ParsedFiveDigits\\2'),
    ('(\D)\d\d\d\d\d\d(\D)', '\\1ParsedSixDigits\\2'),
    ('\d+', 'ParsedDigits')
]

compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]

def generate_preprocessor(replace_patterns):
    compiled_replace_patterns = [(re.compile(p[0]), p[1]) for p in replace_patterns]
    def preprocessor(text):
        # For each pattern, replace it with the appropriate string
        for pattern, replace in compiled_replace_patterns:
            text = re.sub(pattern, replace, text)
        text = text.lower()
        return text
    return preprocessor

generated_patters=generate_preprocessor(replace_patterns)

def preprocessor(text):
    # For each pattern, replace it with the appropriate string
    for pattern, replace in compiled_replace_patterns:
        text = re.sub(pattern, replace, text)
    text = text.lower()
    return text

def _class_frequency(X,y):
    counter = np.zeros(X.shape[1])
    for i, term_freq in enumerate(X.T):
        idx = term_freq.nonzero()[1]
        labels = y[idx]
        counter[i] = len(set(labels))
    return counter

class TFICFTransformer(TfidfTransformer):
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
    def fit(self, X, y=None):
        """Learn the icf vector (global term weights)
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """
        if y is None:
            raise Exception('y can not be None')
        if type(y) is not np.ndarray:
            y = np.array(y)
        n_classes = len(set(y))

        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            cf = _class_frequency(X, y).astype(dtype)

            # perform idf smoothing if required
            cf += int(self.smooth_idf)
            n_classes += int(self.smooth_idf)

            # log+1 instead of log makes sure terms with zero idf don't get
            # suppressed entirely.
            icf = np.log(n_classes / cf) + 1
            self._icf_diag = sp.diags(icf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)
        return self
    def transform(self, X, copy=True):
        """Transform a count matrix to a tf or tf-icf representation
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        if self.sublinear_tf:
            np.log(X.data, X.data)
            X.data += 1

        if self.use_idf:
            check_is_fitted(self, '_icf_diag', 'idf vector is not fitted')

            expected_n_features = self._icf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._icf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X

    @property
    def icf_(self):
        # if _icf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "icf_") is False
        return np.ravel(self._icf_diag.sum(axis=0))

    @icf_.setter
    def icf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._icf_diag = sp.spdiags(value, diags=0, m=n_features,
                                    n=n_features, format='csr')

class TFICFVectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features.

    Equivalent to :class:`CountVectorizer` followed by
    :class:`TficfTransformer`.

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    input : string {'filename', 'file', 'content'}
        If 'filename', the sequence passed as an argument to fit is
        expected to be a list of filenames that need reading to fetch
        the raw content to analyze.

        If 'file', the sequence items must have a 'read' method (file-like
        object) that is called to fetch the bytes in memory.

        Otherwise the input is expected to be the sequence strings or
        bytes items are expected to be analyzed directly.

    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.

    decode_error : {'strict', 'ignore', 'replace'} (default='strict')
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.

    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization
        during the preprocessing step.
        'ascii' is a fast method that only works on characters that have
        an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from
        :func:`unicodedata.normalize`.

    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default=None)
        Override the preprocessing (string transformation) stage while
        preserving the tokenizing and n-grams generation steps.

    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the
        preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside
        word boundaries; n-grams at the edges of words are padded with space.

        If a callable is passed it is used to extract the sequence of features
        out of the raw, unprocessed input.

    stop_words : string {'english'}, list, or None (default=None)
        If a string, it is passed to _check_stop_list and the appropriate stop
        list is returned. 'english' is currently the only supported string
        value.
        There are several known issues with 'english' and you should
        consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).

    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=None)
        If not None, build a vocabulary that only consider the top
        max_features ordered by term frequency across the corpus.

        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are
        indices in the feature matrix, or an iterable over terms. If not
        given, a vocabulary is determined from the input documents.

    binary : boolean (default=False)
        If True, all non-zero term counts are set to 1. This does not mean
        outputs will have only 0/1 values, only that the tf term in tf-idf
        is binary. (Set idf and normalization to False to get 0/1 outputs.)

    dtype : type, optional (default=float64)
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine
        similarity between two vectors is their dot product when l2 norm has
        been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`

    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean (default=True)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean (default=False)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    icf_ : array, shape (n_features)
        The inverse document frequency (IDF) vector; only defined
        if  ``use_idf`` is True.

    stop_words_ : set
        Terms that were ignored because they either:

          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).

        This is only available if no vocabulary was given.

    Examples
    --------
    >>> from sklearn.feature_extraction.text import TficfVectorizer
    >>> corpus = [
    ...     'This is the first document.',
    ...     'This document is the second document.',
    ...     'And this is the third one.',
    ...     'Is this the first document?',
    ... ]
    >>> vectorizer = TficfVectorizer()
    >>> X = vectorizer.fit_transform(corpus)
    >>> print(vectorizer.get_feature_names())
    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    >>> print(X.shape)
    (4, 9)

    See also
    --------
    CountVectorizer : Transforms text into a sparse matrix of n-gram counts.

    TficfTransformer : Performs the TF-IDF transformation from a provided
        matrix of counts.

    Notes
    -----
    The ``stop_words_`` attribute can get large and increase the model size
    when pickling. This attribute is provided only for introspection and can
    be safely removed using delattr or set to None before pickling.
    """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=preprocessor, tokenizer=None, analyzer='word',
                 stopwords=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):

        super(TFICFVectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words = stopwords, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)
        self.stopwords = stopwords

        self._tficf = TFICFTransformer(norm=norm, use_idf=use_idf,
                                       smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    # Broadcast the TF-IDF parameters to the underlying transformer instance
    # for easy grid search and repr

    @property
    def norm(self):
        return self._tficf.norm

    @norm.setter
    def norm(self, value):
        self._tficf.norm = value

    @property
    def use_idf(self):
        return self._tficf.use_idf

    @use_idf.setter
    def use_idf(self, value):
        self._tficf.use_idf = value

    @property
    def smooth_idf(self):
        return self._tficf.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._tficf.smooth_idf = value

    @property
    def sublinear_tf(self):
        return self._tficf.sublinear_tf

    @sublinear_tf.setter
    def sublinear_tf(self, value):
        self._tficf.sublinear_tf = value

    @property
    def icf_(self):
        return self._tficf.icf_

    @icf_.setter
    def icf_(self, value):
        self._validate_vocabulary()
        if hasattr(self, 'vocabulary_'):
            if len(self.vocabulary_) != len(value):
                raise ValueError("idf length = %d must be equal "
                                 "to vocabulary size = %d" %
                                 (len(value), len(self.vocabulary)))
        self._tficf.icf_ = value

    def _check_params(self):
        if self.dtype not in FLOAT_DTYPES:
            warnings.warn("Only {} 'dtype' should be used. {} 'dtype' will "
                          "be converted to np.float64."
                          .format(FLOAT_DTYPES, self.dtype),
                          UserWarning)

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : TficfVectorizer
        """
        self._check_params()
        X = super(TFICFVectorizer, self).fit_transform(raw_documents, y)
        self._tficf.fit(X, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        self._check_params()
        X = super(TFICFVectorizer, self).fit_transform(raw_documents, y)
        self._tficf.fit(X, y)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._tficf.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, '_tficf', 'The tficf vector is not fitted')

        X = super(TFICFVectorizer, self).transform(raw_documents)
        return self._tficf.transform(X, copy=False)

class TFIDFVectorizer(TfidfVectorizer):
    def __init__(self, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stopwords, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)
        self.stopwords = stopwords

class TFVectorizer(CountVectorizer):
    def __init__(self, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stopwords, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)
        self.stopwords = stopwords

# FIX (TODO): Arrumar os parâmetros de inicialização das versões Stemmed dos vectorizer

class StemmedTFICFVectorizer(TFICFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFICFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))

class StemmedTFIDFVectorizer(TFIDFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFIDFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))

class StemmedTFVectorizer(TFVectorizer):
    def __init__(self, lang, stopwords=None, preprocessor=preprocessor, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None, lowercase=True,
                 tokenizer=None, analyzer='word',
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, vocabulary=None, binary=False,
                 dtype=np.float64, norm='l2', use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        super().__init__(
                preprocessor=preprocessor, 
                stop_words=stopwords, input=input, encoding=encoding,
                 decode_error=decode_error, strip_accents=strip_accents, lowercase=lowercase,
                 tokenizer=tokenizer, analyzer=analyzer,
                 token_pattern=token_pattern,
                 ngram_range=ngram_range, max_df=max_df, min_df=min_df,
                 max_features=max_features, vocabulary=vocabulary, binary=binary,
                 dtype=dtype, norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                 sublinear_tf=sublinear_tf)
        self.lang = lang
        self.stopwords = stopwords
        from nltk.stem.snowball import SnowballStemmer
        from nltk import word_tokenize
        self.stemmer = SnowballStemmer(self.lang)
    def build_analyzer(self):
        analyzer = super(TFVectorizer, self).build_analyzer()
        return lambda doc: (self.stemmer.stem(w) for w in analyzer(doc))