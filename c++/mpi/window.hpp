// Copyright (c) 2024 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Authors: Thomas Hahn, Alexander Hampel, Olivier Parcollet, Nils Wentzell

/**
 * @file
 * @brief Provides a C++ wrapper class for an `MPI_Comm` object.
 */

#pragma once

#include "./communicator.hpp"
#include "./macros.hpp"

#include <mpi.h>

#include <span>

namespace mpi {

  template <class BaseType> class shared_window;

  /**
  * @ingroup mpi_essentials
  * @brief A C++ wrapper around `MPI_Win` providing convenient memory window management.
  *
  * @details This class abstracts the complexities of MPI window management, allowing processes 
  *          in an MPI communicator to create and share memory regions efficiently. It supports 
  *          both local buffer-based windows and dynamically allocated memory windows.
  *
  *          If a base pointer is not specified, the constructor will allocate memory internally.
  *
  * @tparam `BaseType` The type of elements stored in the memory window.
  */

  template <class BaseType>
  class window {
    friend class shared_window<BaseType>;
    MPI_Win win{MPI_WIN_NULL};
    bool is_owned{true};
    std::span<BaseType> view;
  public:

    window() = default;
    window(window const&) = delete;
    window(window &&other) noexcept : win{std::exchange(other.win, MPI_WIN_NULL)},
                                      is_owned{std::exchange(other.is_owned, true)},
                                      view{std::exchange(other.view, std::span<BaseType>())} {}
    window& operator=(window const&) = delete;
    window& operator=(window &&rhs) noexcept {
      if (this != std::addressof(rhs)) {
        this->free();
        this->win = std::exchange(rhs.win, MPI_WIN_NULL);
        this->view = std::exchange(rhs.view, std::span<BaseType>());
        this->is_owned = std::exchange(rhs.is_owned, true);
      }
      return *this;
    }

    /**
    * @brief Constructs an MPI window over an existing local memory buffer.
    *
    * @details This constructor allows creating a window using a pre-allocated memory buffer.
    *          The window provides access to the specified memory region across MPI processes
    *          within the given communicator. The buffer is not freed upon destruction unless
    *          explicitly owned by the instance.
    *
    * @param c The MPI communicator that defines the group of processes sharing the window.
    * @param base Pointer to the base address of the memory buffer.
    * @param size The number of elements of type `BaseType` in the buffer set to 0 if not defined.
    * @param info Additional MPI information (default: `MPI_INFO_NULL`).
    */

    explicit window(communicator &c, BaseType *base, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept(false) {
      ASSERT(size >= 0)
      ASSERT(!(base == nullptr && size > 0))
      if (has_env) {
        MPI_Win_create(base, size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &win);
        view = std::span<BaseType>(base, size);
      } else {
        is_owned = false;
        view = std::span<BaseType>(base, size);
      }
    }

    /**
    * @brief Constructs an MPI window with dynamically allocated memory.
    *
    * @details This constructor allocates a new memory buffer locally and creates an MPI window 
    *          over it. The allocated memory is automatically freed when the window is destroyed.
    *          This is useful when the memory region is meant to be shared across processes
    *          without needing an external buffer.
    *
    * @param c The MPI communicator that defines the group of processes sharing the window.
    * @param size The number of elements of type `BaseType` to allocate.
    * @param info Additional MPI information (default: `MPI_INFO_NULL`).
    */
    explicit window(communicator &c, MPI_Aint size = 0, MPI_Info info = MPI_INFO_NULL) noexcept {
      ASSERT(size >= 0)
      if (has_env) {
        void *baseptr = nullptr;
        MPI_Win_allocate(size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &baseptr, &win);
        view = std::span<BaseType>(static_cast<BaseType*>(baseptr), size);
      } else {
        is_owned = true;
        BaseType* baseptr = new BaseType[size];
        view = std::span<BaseType>(baseptr, size);
      }
    }

    /**
    * @brief Destroys the window and releases allocated resources.
    *
    * @details If the window owns an allocated memory buffer, it will be automatically freed.
    *          Otherwise, only the MPI window handle is released.
    */
    ~window() { free(); }

    explicit operator MPI_Win() const noexcept { return win; };
    explicit operator MPI_Win*() noexcept { return &win; };

    void free() noexcept {
      if (has_env) {
        if (win != MPI_WIN_NULL) {
          MPI_Win_free(&win);
          // Mention data is only freed in non MPI env
        }
      } else {
        if (is_owned) {
          delete[] view.data();
        }
        view = std::span<BaseType>();
      }
    }

    /**
    * @brief Synchronizes all RMA operations within an access epoch.
    *
    * @details This function acts as a barrier for remote memory access (RMA) operations,
    *          ensuring all previous operations on the window are completed before continuing.
    *
    * @param assert An assertion flag that provides optimization hints to MPI (default: 0).
    */
    void fence(int assert = 0) const noexcept {
      if (has_env) {
        MPI_Win_fence(assert, win);
      }
    }

    /**
    * @brief Ensures completion of all outstanding RMA operations.
    *
    * @details This function forces all RMA operations issued to a specific rank (or all ranks)
    *          to complete at both the origin and the target before proceeding.
    *
    * @param rank The rank to flush operations for. If negative or no rank is specified, flushes all ranks .
    */
    void flush(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_flush_all(win);
        } else {
          MPI_Win_flush(rank, win);
        }
      }
    }

    /**
    * @brief Synchronizes the public and private copies of the window.
    *
    * @details Ensures that any updates to the local memory are visible in the public window 
    *          and vice versa. This is particularly useful when working with shared memory.
    */
    void sync() const noexcept {
      if (has_env) {
        MPI_Win_sync(win);
      }
    }

    /**
    * @brief Starts an RMA access epoch.
    *
    * @details Locks access to the memory window for a specific rank or all ranks,
    *          preventing concurrent modifications. This enables fine-grained control 
    *          over remote memory access operations.
    *
    * @param rank The rank to lock access for. If negative, locks all ranks (default: -1).
    * @param lock_type The type of lock (e.g., `MPI_LOCK_SHARED` or `MPI_LOCK_EXCLUSIVE`).
    * @param assert An assertion flag providing optimization hints to MPI (default: 0).
    */
    void lock(int rank = -1, int lock_type = MPI_LOCK_SHARED, int assert = 0) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_lock_all(assert, win);
        } else {
          MPI_Win_lock(lock_type, rank, assert, win);
        }
      }
    }

    /**
    * @brief Completes an RMA access epoch started by `lock()`.
    *
    * @details Unlocks access to the memory window for a specific rank or all ranks,
    *          allowing other processes to access or modify the window.
    *
    * @param rank The rank to unlock access for. If negative, unlocks all ranks (default: -1).
    */
    void unlock(int rank = -1) const noexcept {
      if (has_env) {
        if (rank < 0) {
          MPI_Win_unlock_all(win);
        } else {
          MPI_Win_unlock(rank, win);
        }
      }
    }

    /**
    * @brief Reads data from a remote memory window.
    *
    * @details This function retrieves data from a remote process's memory window and 
    *          stores it in the local buffer. It supports different data types via templates.
    *
    * @tparam `TargetType` The data type at the target memory (defaults to BaseType).
    * @tparam `OriginType` The data type at the origin memory.
    * @param origin_addr Pointer to the local memory buffer where the data will be stored.
    * @param origin_count Number of elements to retrieve.
    * @param target_rank Rank of the target process from which data is fetched.
    * @param target_disp Displacement (in elements) from the start of the target memory window.
    * @param target_count Number of elements to read from the target. If negative, defaults to `origin_count`.
    */
    template <typename TargetType = BaseType, typename OriginType>
    requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void get(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Get(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        // What would happen if target_disp is negative
        if (target_rank != 0) {
          return;
        }

        std::span<OriginType> origin(origin_addr, origin_count);
        auto target_begin = view.begin();
        std::advance(target_begin, target_disp);
        auto target_end = target_begin;
        std::advance(target_end, target_count_);
        std::copy(target_begin, target_end, origin.begin());
      }
    }

    /**
    * @brief Writes data to a remote memory window.
    *
    * @details This function transfers data from a local buffer to a remote process's 
    *          memory window. It supports different data types via templates.
    *
    * @tparam `TargetType` The data type at the target memory (defaults to BaseType).
    * @tparam `OriginType` The data type at the origin memory.
    * @param origin_addr Pointer to the local memory buffer containing the data to be sent.
    * @param origin_count Number of elements to transfer.
    * @param target_rank Rank of the target process to which data is written.
    * @param target_disp Displacement (in elements) from the start of the target memory window.
    * @param target_count Number of elements to write to the target. If negative, defaults to `origin_count`.
    */
    template <typename TargetType = BaseType, typename OriginType>
    requires(has_mpi_type<OriginType> && has_mpi_type<TargetType>)
    void put(OriginType *origin_addr, int origin_count, int target_rank, MPI_Aint target_disp = 0, int target_count = -1) const noexcept {
      int target_count_ = target_count < 0 ? origin_count : target_count;
      if (has_env) {
        MPI_Datatype origin_datatype = mpi_type<OriginType>::get();
        MPI_Datatype target_datatype = mpi_type<TargetType>::get();
        MPI_Put(origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count_, target_datatype, win);
      } else {
        if (target_rank != 0) {
          return;
        }

        std::span<OriginType> origin(origin_addr, origin_count);
        auto target_begin = view.begin();
        std::advance(target_begin, target_disp);
        std::copy(origin.begin(), origin.end(), target_begin);
      }
    }

    /**
    * @brief Retrieves the value of a window attribute.
    *
    * @details This function queries an attribute associated with an MPI window,
    *          such as memory model or lock type. If the attribute exists, it returns 
    *          a pointer to its value; otherwise, it returns `nullptr`.
    *
    * @param win_keyval The key identifying the attribute.
    * @return A pointer to the attribute value, or `nullptr` if not found.
    */
    void* get_attr(int win_keyval) const noexcept {
      if (has_env) {
        int flag;
        void *attribute_val;
        MPI_Win_get_attr(win, win_keyval, &attribute_val, &flag);
        ASSERT(flag)
        return attribute_val;
      } else {
        ASSERT(has_env)
        return nullptr;
      }
    }

    /**
    * @brief Retrieves the base address of the memory window.
    *
    * @details This function returns a pointer to the base address of the memory associated with the MPI window.
    *          If MPI is enabled (`has_env` is true), it queries the `MPI_WIN_BASE` attribute.
    *          If MPI is not enabled, it returns the base address of the local memory view.
    *
    * @return A pointer to the base address of the window memory, or `nullptr` if the window is invalid.
    */
    BaseType* base() const noexcept {
      if (has_env) {
        if (win == MPI_WIN_NULL) {
          return nullptr;
        }
        return static_cast<BaseType*>(get_attr(MPI_WIN_BASE));
      } else {
        return view.data();
      }
    }

    /**
    * @brief Retrieves the size of the memory window.
    *
    * @details This function returns the total size (in bytes) of the memory associated with the MPI window.
    *          If MPI is enabled (`has_env` is true), it queries the `MPI_WIN_SIZE` attribute.
    *          Otherwise, it returns the size of the local view in bytes.
    *
    * @return The size of the MPI window in bytes.
    */

    MPI_Aint size() const noexcept {
      if (has_env) {
        return *static_cast<MPI_Aint*>(get_attr(MPI_WIN_SIZE));
      } else {
        return view.size_bytes();
      }
    }

    /**
    * @brief Retrieves the displacement unit of the memory window.
    *
    * @details The displacement unit determines the scaling factor for address displacements 
    *          in the MPI window. If MPI is enabled (`has_env` is true), it queries the `MPI_WIN_DISP_UNIT` attribute.
    *          Otherwise, it returns the size of a single element in the local view.
    *
    * @return The displacement unit (in bytes).
    */

    int disp_unit() const noexcept {
      if (has_env) {
        return *static_cast<int*>(get_attr(MPI_WIN_DISP_UNIT));
      } else {
        return sizeof(decltype(view)::element_type);
      }
    }

    std::span<BaseType>& data() noexcept {
      return view;
    }
    std::span<BaseType>& data() const noexcept {
      return view;
    }
  };

  /**
  * @brief A shared memory window abstraction using MPI.
  *
  * @details This class provides an interface for creating and managing an MPI shared memory window.
  *          When MPI is enabled, memory is allocated using `MPI_Win_allocate_shared`. Otherwise,
  *          a standard dynamic allocation is performed.
  *
  * @tparam BaseType The data type stored in the shared memory window.
  */
  template <class BaseType>
  class shared_window : public window<BaseType> {
  public:
    /// Default constructor
    shared_window() = default;

    /**
     * @brief Constructs a shared memory window.
     *
     * @details If MPI is enabled, this constructor allocates shared memory within the given communicator.
     *          Otherwise, it falls back to a standard dynamic allocation.
     *
     * @param c The shared communicator.
     * @param size The number of elements of type `BaseType` to allocate.
     * @param info MPI_Info object for optimization hints (defaults to `MPI_INFO_NULL`).
     */
    explicit shared_window(shared_communicator& c, MPI_Aint size, MPI_Info info = MPI_INFO_NULL) noexcept {
      ASSERT(size >= 0)
      if (has_env) {
        void* baseptr = nullptr;
        MPI_Win_allocate_shared(size * sizeof(BaseType), sizeof(BaseType), info, c.get(), &baseptr, &(this->win));
        this->view = std::span<BaseType>(static_cast<BaseType*>(baseptr), size);
      } else {
        this->is_owned = true;
        BaseType* baseptr = new BaseType[size];
        this->view = std::span<BaseType>(baseptr, size);
      }
    }

    /**
     * @brief Queries attributes of a shared memory window.
     *
     * @details Retrieves the size, displacement unit, and base address of the shared memory region for a given rank.
     *          In non-MPI mode, it returns attributes of the local `std::span<BaseType>`.
     *
     * @param rank The rank within the communicator (defaults to `MPI_PROC_NULL` for querying all ranks).
     * @return A tuple containing (size in bytes, displacement unit, base pointer).
     */
    std::tuple<MPI_Aint, int, void*> query(int rank = MPI_PROC_NULL) const noexcept {
      if (has_env) {
        MPI_Aint size = 0;
        int disp_unit = 0;
        void *baseptr = nullptr;
        MPI_Win_shared_query(this->win, rank, &size, &disp_unit, &baseptr);
        return {size, disp_unit, baseptr};
      } else {
        return {this->view.size_bytes(), sizeof(BaseType), this->view.data()};
      }
    }

    // Override the commonly used attributes of the window base class

    /**
     * @brief Returns the base address of the shared memory for a specific rank.
     *
     * @param rank The rank whose base address should be retrieved (defaults to `MPI_PROC_NULL`).
     * @return A pointer to the base address.
     */
    BaseType* base(int rank = MPI_PROC_NULL) const noexcept {
      return static_cast<BaseType*>(std::get<2>(query(rank)));
    }

    /**
     * @brief Returns the number of elements stored in the shared memory window.
     *
     * @param rank The rank whose memory size should be retrieved (defaults to `MPI_PROC_NULL`).
     * @return The number of elements in the shared window.
     */
    MPI_Aint size(int rank = MPI_PROC_NULL) const noexcept {
      return std::get<0>(query(rank)) / sizeof(BaseType);
    }

    /**
     * @brief Returns the displacement unit of the shared memory.
     *
     * @param rank The rank whose displacement unit should be retrieved (defaults to `MPI_PROC_NULL`).
     * @return The displacement unit (usually equal to `sizeof(BaseType)`).
     */
    int disp_unit(int rank = MPI_PROC_NULL) const noexcept {
      return std::get<1>(query(rank));
    }
  };

};
