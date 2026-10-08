
#ifndef CIFPP_EXPORT_H
#define CIFPP_EXPORT_H

#ifdef CIFPP_STATIC_DEFINE
#  define CIFPP_EXPORT
#  define CIFPP_NO_EXPORT
#else
#  ifndef CIFPP_EXPORT
#    ifdef cifpp_EXPORTS
        /* We are building this library */
#      define CIFPP_EXPORT 
#    else
        /* We are using this library */
#      define CIFPP_EXPORT 
#    endif
#  endif

#  ifndef CIFPP_NO_EXPORT
#    define CIFPP_NO_EXPORT 
#  endif
#endif

#ifndef CIFPP_DEPRECATED
#  define CIFPP_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef CIFPP_DEPRECATED_EXPORT
#  define CIFPP_DEPRECATED_EXPORT CIFPP_EXPORT CIFPP_DEPRECATED
#endif

#ifndef CIFPP_DEPRECATED_NO_EXPORT
#  define CIFPP_DEPRECATED_NO_EXPORT CIFPP_NO_EXPORT CIFPP_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef CIFPP_NO_DEPRECATED
#    define CIFPP_NO_DEPRECATED
#  endif
#endif

#endif /* CIFPP_EXPORT_H */
